import os
import sys
import ast
import asyncio
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm_vertex import GoogleVertexLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import EndFrame, ErrorFrame, TranscriptionFrame, LLMFullResponseStartFrame, TTSAudioRawFrame
from pipecat.transcriptions.language import Language
from pipecat.turns.mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy
from services.supabase_service import update_lead_status

import json
import time
import aiohttp
from deepgram import LiveOptions
from google.oauth2 import service_account
from google.cloud import texttospeech_v1

_GOOGLE_TTS_VOICE_NAMES = None
_PERF = {"last_user_text_ts": None, "last_llm_start_ts": None, "tts_first_audio_logged": False}


def build_deepgram_live_options(*, encoding: str, sample_rate: int, language: str) -> LiveOptions:
    language = (language or "ar").strip()
    model = (os.getenv("DEEPGRAM_MODEL") or "").strip() or None
    if not model:
        model = "nova-2-phonecall" if language.startswith("en") else "nova-2"

    smart_format = (os.getenv("DEEPGRAM_SMART_FORMAT") or "true").strip().lower() in {"1", "true", "yes", "y"}
    punctuate = (os.getenv("DEEPGRAM_PUNCTUATE") or "false").strip().lower() in {"1", "true", "yes", "y"}
    profanity_filter = (os.getenv("DEEPGRAM_PROFANITY_FILTER") or "false").strip().lower() in {"1", "true", "yes", "y"}

    utterance_end_ms = "1000"
    utterance_end_ms_env = (os.getenv("DEEPGRAM_UTTERANCE_END_MS") or "").strip()
    if utterance_end_ms_env:
        if utterance_end_ms_env.lower() in {"0", "false", "off", "none", "null"}:
            utterance_end_ms = None
        else:
            try:
                utterance_end_ms_int = int(float(utterance_end_ms_env))
                utterance_end_ms = str(utterance_end_ms_int)
            except Exception:
                utterance_end_ms = "1000"

    logger.info(
        f"Deepgram options: model={model} language={language} encoding={encoding} sample_rate={sample_rate} "
        f"smart_format={smart_format} punctuate={punctuate} profanity_filter={profanity_filter} "
        f"utterance_end_ms={utterance_end_ms}"
    )

    live_options_kwargs = dict(
        encoding=encoding,
        language=language,
        model=model,
        channels=1,
        sample_rate=sample_rate,
        interim_results=True,
        smart_format=smart_format,
        punctuate=punctuate,
        profanity_filter=profanity_filter,
        vad_events=False,
    )
    if utterance_end_ms is not None:
        live_options_kwargs["utterance_end_ms"] = utterance_end_ms
    return LiveOptions(**live_options_kwargs)


def resolve_stt_language(language: str) -> Language:
    normalized = (language or "").strip().lower()
    if normalized.startswith("ar"):
        return Language.AR
    if normalized.startswith("en"):
        return Language.EN
    return Language.AR


async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id:
        return
    telnyx_key = os.getenv("TELNYX_API_KEY")
    if not telnyx_key:
        logger.warning("TELNYX_API_KEY missing; cannot hang up call.")
        return
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup"
    headers = {"Authorization": f"Bearer {telnyx_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"reason": "normal_clearing"}) as resp:
                if resp.status >= 300:
                    text = await resp.text()
                    logger.warning(f"Telnyx hangup failed: {resp.status} {text}")
    except Exception as e:
        logger.warning(f"Telnyx hangup exception: {e}")


class STTPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            _PERF["last_user_text_ts"] = time.monotonic()
            _PERF["last_llm_start_ts"] = None
            _PERF["tts_first_audio_logged"] = False
        await self.push_frame(frame, direction)


class LLMPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            now = time.monotonic()
            _PERF["last_llm_start_ts"] = now
            user_ts = _PERF.get("last_user_text_ts")
            if user_ts is not None:
                logger.info(f"Latency user_text→llm_start={now - user_ts:.3f}s")
        await self.push_frame(frame, direction)


class TTSPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and not _PERF.get("tts_first_audio_logged"):
            now = time.monotonic()
            llm_ts = _PERF.get("last_llm_start_ts")
            user_ts = _PERF.get("last_user_text_ts")
            if llm_ts is not None:
                logger.info(f"Latency llm_start→tts_audio={now - llm_ts:.3f}s")
            if user_ts is not None:
                logger.info(f"Latency user_text→tts_audio={now - user_ts:.3f}s")
            _PERF["tts_first_audio_logged"] = True
        await self.push_frame(frame, direction)


class STTFailureFallback(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._handled = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame) and not self._handled:
            error_text = str(getattr(frame, "error", ""))
            if "Unable to connect to Deepgram" in error_text:
                self._handled = True
                await self.push_frame(TTSSpeakFrame("صار في مشكلة بالصوت، معك حق. لحظة وبنرجع."), direction)
                await self.push_frame(EndFrame(), direction)
        await self.push_frame(frame, direction)

# Helper function to get Google Credentials
def get_google_credentials():
    # 1. Try File Path First (Preferred)
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path and os.path.exists(path):
        logger.info(f"Found Google Credentials file at: {path}")
        return None # Return None so run_bot uses the path directly

    # 2. Try JSON String (Fallback)
    json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if json_str:
        logger.info("Loading Google Credentials from JSON string env var")
        
        # Basic cleanup of wrapping quotes
        json_str = json_str.strip().strip("'").strip('"')
        
        # Handle escaped quotes commonly found in env vars (e.g. \"type\" -> "type")
        json_str = json_str.replace('\\"', '"')
        
        # Handle boolean values if they are Python-style
        json_str = json_str.replace("True", "true").replace("False", "false")

        try:
            # Try parsing as standard JSON
            info = json.loads(json_str)
            logger.info("Successfully parsed credentials as JSON")
        except json.JSONDecodeError:
            # Fallback: Try parsing as a Python dictionary string
            try:
                logger.info("JSON decode failed, trying ast.literal_eval")
                info = ast.literal_eval(json_str)
            except (ValueError, SyntaxError) as e:
                logger.error(f"Failed to parse Google Credentials: {e}")
                logger.error(f"First 100 chars: {json_str[:100]}")
                return None
        
        # Post-processing: Handle private key newlines if needed
        if "private_key" in info:
            # Ensure we have actual newlines, not escaped ones
            info["private_key"] = info["private_key"].replace("\\n", "\n")
            
        return info
    
    return None

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot for lead: {lead_data['id']}")

    # 0. Handle Telnyx Handshake to get stream_id
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU"
    stream_sample_rate = 8000
    try:
        # Telnyx typically sends a JSON payload first with event="connected"
        # Then it sends event="start" which contains the stream_id
        # We need to loop until we find the stream_id or a reasonable timeout/limit
        
        logger.info("Waiting for Telnyx 'start' event with stream_id...")
        
        for _ in range(3): # Try up to 3 messages
            msg_text = await websocket_client.receive_text()
            logger.info(f"Received Telnyx message: {msg_text}")
            msg = json.loads(msg_text)
            
            # Check if this message has media_format information (usually in 'start' event)
            if "media_format" in msg:
                 encoding = msg["media_format"].get("encoding", "").upper()
                 stream_sample_rate = int(msg["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                 logger.info(f"Telnyx Media Format (direct): {msg['media_format']}")
                 if encoding == "G729":
                     logger.error("CRITICAL: Telnyx is sending G.729 audio. Pipecat requires PCMU (G.711u) or PCMA (G.711a).")
                     logger.error("Please disable G.729 in your Telnyx Portal SIP Connection settings.")
                 elif encoding == "PCMA":
                     logger.info("Detected PCMA encoding, updating serializer.")
                     inbound_encoding = "PCMA"
                 elif encoding == "PCMU":
                     inbound_encoding = "PCMU"
                 elif encoding == "L16":
                     inbound_encoding = "L16"

            elif "start" in msg and "media_format" in msg["start"]:
                 encoding = msg["start"]["media_format"].get("encoding", "").upper()
                 stream_sample_rate = int(msg["start"]["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                 logger.info(f"Telnyx Media Format (nested in start): {msg['start']['media_format']}")
                 if encoding == "G729":
                     logger.error("CRITICAL: Telnyx is sending G.729 audio. Pipecat requires PCMU (G.711u) or PCMA (G.711a).")
                     logger.error("Please disable G.729 in your Telnyx Portal SIP Connection settings.")
                 elif encoding == "PCMA":
                     logger.info("Detected PCMA encoding, updating serializer.")
                     inbound_encoding = "PCMA"
                 elif encoding == "PCMU":
                     inbound_encoding = "PCMU"
                 elif encoding == "L16":
                     inbound_encoding = "L16"

            # Check standard locations for stream_id
            if "stream_id" in msg:
                stream_id = msg["stream_id"]
                logger.info(f"Captured stream_id (direct): {stream_id}")
                break
            elif "data" in msg and "stream_id" in msg["data"]:
                stream_id = msg["data"]["stream_id"]
                logger.info(f"Captured stream_id (in data): {stream_id}")
                break
            elif msg.get("event") == "start":
                 # capture stream_id if present
                 if "stream_id" in msg:
                      stream_id = msg["stream_id"]
                      logger.info(f"Captured stream_id (from start event): {stream_id}")
                      break
            
            # If we didn't find it, we loop again to get the next message
            
        if stream_id == "telnyx_stream_placeholder":
             logger.warning("Could not find stream_id in initial messages, using placeholder.")
             
    except Exception as e:
        logger.error(f"Failed to capture stream_id from initial message: {e}")

    # Get Credentials (dictionary)
    google_creds_dict = get_google_credentials()
    
    # Prepare Credentials for different services
    google_creds_str = None
    google_creds_obj = None

    if google_creds_dict:
        # LLM needs JSON string
        google_creds_str = json.dumps(google_creds_dict)
        # TTS likely needs Credentials object
        google_creds_obj = service_account.Credentials.from_service_account_info(google_creds_dict)
    
    # 1. Services
    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.0, start_secs=0.15, stop_secs=0.4, confidence=0.45))
    call_end_delay_s = 2.5
    try:
        call_end_delay_s = float(os.getenv("CALL_END_DELAY_S") or 2.5)
    except Exception:
        call_end_delay_s = 2.5
    
    stt_language = os.getenv("DEEPGRAM_LANGUAGE") or "ar"
    stt_provider_env = (os.getenv("STT_PROVIDER") or "").strip().lower()
    stt_provider = stt_provider_env or ("google" if stt_language.lower().startswith("ar") else "deepgram")

    if stt_provider == "deepgram" and stt_language.lower().startswith("ar"):
        logger.warning("Deepgram streaming does not support Arabic. Switching STT provider to Google.")
        stt_provider = "google"

    if stt_provider == "deepgram":
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_key:
            logger.error("CRITICAL: DEEPGRAM_API_KEY is missing in environment variables!")
        else:
            logger.info(f"Deepgram API Key found (len={len(deepgram_key)}).")

        dg_encoding = "mulaw" if inbound_encoding == "PCMU" else "alaw" if inbound_encoding == "PCMA" else "linear16"
        logger.info(f"Initializing Deepgram with encoding: {dg_encoding} (inbound: {inbound_encoding})")

        stt_live_options = build_deepgram_live_options(
            encoding=dg_encoding,
            sample_rate=stream_sample_rate,
            language=stt_language,
        )
        stt = DeepgramSTTService(
            api_key=deepgram_key,
            live_options=stt_live_options,
        )
    else:
        stt_language_enum = resolve_stt_language(stt_language)
        stt_model = os.getenv("GOOGLE_STT_MODEL") or ("latest_short" if stt_language_enum == Language.AR else "latest_long")
        stt_location = os.getenv("GOOGLE_STT_LOCATION") or "global"
        logger.info(f"Using Google STT model: {stt_model} ({stt_location})")
        stt_params = GoogleSTTService.InputParams(
            languages=[stt_language_enum],
            model=stt_model,
            enable_automatic_punctuation=True,
            enable_interim_results=True,
            profanity_filter=False,
            enable_voice_activity_events=False,
        )
        stt_kwargs = {
            "sample_rate": stream_sample_rate,
            "location": stt_location,
            "params": stt_params,
        }
        if google_creds_str:
            stt_kwargs["credentials"] = google_creds_str
        else:
            stt_kwargs["credentials_path"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        stt = GoogleSTTService(**stt_kwargs)
    
    # 2. Tools
    tools = [
        {
            "function_declarations": [
                {
                    "name": "update_lead_status_confirmed",
                    "description": "Call this when the customer confirms the order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Reason for confirmation"}
                        },
                    },
                },
                {
                    "name": "update_lead_status_cancelled",
                    "description": "Call this when the customer cancels the order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Reason for cancellation"}
                        },
                    },
                },
            ]
        }
    ]

    # Initialize LLM with either path or credentials object
    llm_model = os.getenv("GOOGLE_VERTEX_MODEL") or "gemini-2.0-flash-001"
    logger.info(f"Using Vertex model: {llm_model}")
    llm_kwargs = {
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "location": "us-central1",
        "model": llm_model,
        "tools": tools
    }
    
    if google_creds_str:
        llm_kwargs["credentials"] = google_creds_str
    else:
        llm_kwargs["credentials_path"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    llm = GoogleVertexLLMService(**llm_kwargs)
    
    tts_encoding = "mulaw" if inbound_encoding == "PCMU" else "alaw" if inbound_encoding == "PCMA" else "linear16"
    tts_sample_rate = stream_sample_rate
    voice_id = os.getenv("GOOGLE_TTS_VOICE_ID") or "ar-XA-Chirp3-HD-Charon"
    fallback_voice_ids = ["ar-XA-Chirp3-HD-Charon", "ar-XA-Chirp3-HD-Aoede"]
    global _GOOGLE_TTS_VOICE_NAMES
    try:
        if _GOOGLE_TTS_VOICE_NAMES is None:
            client = (
                texttospeech_v1.TextToSpeechClient(credentials=google_creds_obj)
                if google_creds_obj
                else texttospeech_v1.TextToSpeechClient()
            )
            _GOOGLE_TTS_VOICE_NAMES = {v.name for v in client.list_voices().voices}
        if "Chirp3-HD" not in voice_id:
            locale = "ar-XA"
            candidates = [v for v in _GOOGLE_TTS_VOICE_NAMES if v.startswith(f"{locale}-Chirp3-HD-")]
            if candidates:
                logger.warning(f"Non-Chirp voice configured for streaming TTS ({voice_id}); switching to {candidates[0]}.")
                voice_id = candidates[0]
        if voice_id not in _GOOGLE_TTS_VOICE_NAMES:
            fallback_voice_id = next((v for v in fallback_voice_ids if v in _GOOGLE_TTS_VOICE_NAMES), None)
            if fallback_voice_id:
                logger.warning(f"Google TTS voice not found: {voice_id}. Falling back to {fallback_voice_id}.")
                voice_id = fallback_voice_id
            else:
                locale = "ar-XA"
                candidates = [v for v in _GOOGLE_TTS_VOICE_NAMES if v.startswith(f"{locale}-Chirp3-HD-")]
                if candidates:
                    logger.warning(f"Google TTS voice not found: {voice_id}. Falling back to {candidates[0]}.")
                    voice_id = candidates[0]
    except Exception as e:
        if voice_id.startswith("ar-JO"):
            fallback_voice_id = fallback_voice_ids[0]
            logger.warning(f"Google TTS voice {voice_id} likely invalid. Falling back to {fallback_voice_id}. ({e})")
            voice_id = fallback_voice_id
    speaking_rate = None
    try:
        speaking_rate_env = os.getenv("GOOGLE_TTS_SPEAKING_RATE")
        if speaking_rate_env:
            speaking_rate = float(speaking_rate_env)
        else:
            speaking_rate = 1.05
    except Exception:
        speaking_rate = 1.05
    tts_kwargs = {
        "voice_id": voice_id,
        "sample_rate": tts_sample_rate,
        "encoding": tts_encoding,
        "params": GoogleTTSService.InputParams(language=Language.AR, speaking_rate=speaking_rate)
    }
    if google_creds_obj:
        tts_kwargs["credentials"] = google_creds_obj
    else:
        tts_kwargs["credentials_path"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    tts = GoogleTTSService(**tts_kwargs)

    # 3. Handlers
    async def confirm_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Confirming order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CONFIRMED')
        await result_callback("Order confirmed successfully.")
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))
        else:
            logger.warning("No call_control_id; cannot hang up after confirmation.")

    async def cancel_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Cancelling order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CANCELLED')
        await result_callback("Order cancelled.")
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))
        else:
            logger.warning("No call_control_id; cannot hang up after cancellation.")

    llm.register_function(
        "update_lead_status_confirmed",
        confirm_order
    )
    llm.register_function(
        "update_lead_status_cancelled",
        cancel_order
    )

    # 3. Transport (Telnyx)
    # Note: TelnyxFrameSerializer requires stream_id and call_control_id for full functionality (like hanging up)
    # In a basic setup, we might not have them immediately from the websocket handshake in the same way.
    # We will instantiate it with placeholders or extract if possible.
    # For now, we'll use a dummy stream_id if we don't have one, or rely on what Pipecat needs.
    
    # We need to know the call_control_id to hang up. 
    # If we can't get it from the websocket, we might need to pass it from the webhook handler -> main.py -> run_bot
    # But for now, let's assume we just want audio streaming working.
    
    serializer = TelnyxFrameSerializer(
        stream_id=stream_id, # Captured from handshake
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding=inbound_encoding,
        inbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(
            sample_rate=stream_sample_rate
        )
    )
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            session_timeout=300,
            vad_enabled=True,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=stream_sample_rate,
            audio_out_sample_rate=stream_sample_rate
        )
    )

    # 4. Prompt
    system_prompt = f"""
# IDENTITY
You are "Khalid", a professional, polite, and native Jordanian delivery coordinator from "Kawkab Delivery" (شركة كوكب للتوصيل).
Your goal is to confirm delivery details with customers in a way that feels 100% human and local to Amman, Jordan.

# PERSONALITY & TONE
- **Voice**: Friendly, energetic, and "Ibn Nas" (well-mannered).
- **Style**: Use the Ammani/Urban Jordanian dialect.
- **Efficiency**: Jordanians appreciate quick calls. Keep responses under 10 words unless explaining something.
- **Rules**: Never speak Formal Arabic (Fusha). Never speak English unless the customer starts in English.

# MANDATORY OPENING (First Phrase)
Your very first sentence must be EXACTLY:
"السلام عليكم، معك خالد من شركة كوكب للتوصيل. كيفك يا {lead_data['customer_name']}؟ في إلك معنا طلب {lead_data['order_items']} المفروض يوصلك على الساعة {lead_data['delivery_time']}. بس حبيت أتأكد إذا الأمور تمام ونبعتلك السائق؟"

# DIALECT GUIDELINES
- Use "G" for "Qaf" (e.g., 'Galleh' for 'Qalleh').
- Use local fillers: "يا هلا والله", "أبشر", "من عيوني", "عشان هيك", "هسا", "مية مية".
- If they say "Yes/Okay": Respond with "ممتاز، أبشر" or "مية مية، هسا برتب مع الشوفير".
- If they say "No/Cancel": Respond with "ولا يهمك، حصل خير. بس بقدر أعرف شو السبب للإلغاء؟".

# LOGIC & TOOLS
1. **Confirmation**: If they agree, call `update_lead_status_confirmed` and say: "تمام، هسا بنرتب الأمور ويوصلك على الموعد إن شاء الله. مع السلامة."
2. **Cancellation**: If they cancel, call `update_lead_status_cancelled` and say: "تم، لغينا الطلب. يومك سعيد، مع السلامة."
3. **Handling Interruption**: If the user says "Hello?" or "Are you there?" (ألو / معك؟ / وينك؟), respond immediately with: "معك معك، تفضل..." and do NOT repeat the full introduction.

# CONTEXT
- Customer: {lead_data['customer_name']}
- Items: {lead_data['order_items']}
- Time: {lead_data['delivery_time']}
"""

    # 4. Prompt & Context
    # Single context with system prompt AND initial user trigger
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    context = LLMContext(messages=messages)

    user_turn_strategies = UserTurnStrategies(
        start=[TranscriptionUserTurnStartStrategy(use_interim=False)],
        stop=[TranscriptionUserTurnStopStrategy(timeout=0.4)],
    )

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=user_turn_strategies,
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            user_turn_stop_timeout=0.6,
        ),
    )

    stt_perf = STTPerf()
    llm_perf = LLMPerf()
    tts_perf = TTSPerf()

    pipeline = Pipeline([
        transport.input(),
        stt,
        STTFailureFallback(),
        stt_perf,
        aggregators.user(),
        llm,
        llm_perf,
        tts,
        tts_perf,
        transport.output(),
        aggregators.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    customer_name = lead_data.get("customer_name") or "العميل"
    order_items = lead_data.get("order_items") or ""
    delivery_time = lead_data.get("delivery_time") or ""
    greeting_text = f"السلام عليكم، معك خالد من شركة كوكب للتوصيل. كيفك يا {customer_name}؟ في إلك معنا طلب {order_items} المفروض يوصلك على الساعة {delivery_time}. بس حبيت أتأكد إذا الأمور تمام ونبعتلك السائق؟"

    logger.info(f"Queuing greeting (inbound_encoding={inbound_encoding}, tts_encoding={tts_encoding})")
    await task.queue_frames([TTSSpeakFrame(greeting_text)])
    logger.info("Starting single pipeline run")
    await runner.run(task)
