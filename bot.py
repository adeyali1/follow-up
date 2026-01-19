import os
import sys
import ast
import asyncio
import re
import aiohttp
from urllib.parse import quote
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm_vertex import GoogleVertexLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, TranscriptionFrame, LLMFullResponseStartFrame, TTSAudioRawFrame
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy
from services.supabase_service import update_lead_status

import json
import time
from google.oauth2 import service_account
from google.cloud import texttospeech_v1
from deepgram import LiveOptions
from pipecat.audio.utils import create_stream_resampler

_GOOGLE_TTS_VOICE_NAMES = None
_PERF = {"last_user_text_ts": None, "last_llm_start_ts": None, "tts_first_audio_logged": False}
_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None}


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


class MultimodalPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "TranscriptionFrame":
            try:
                if getattr(frame, "role", None) == "user":
                    _MM["last_user_transcription_ts"] = time.monotonic()
            except Exception:
                pass
        if frame_name in {"BotStartedSpeakingFrame", "TTSStartedFrame"}:
            now = time.monotonic()
            _MM["last_bot_started_ts"] = now
            user_ts = _MM.get("last_user_transcription_ts")
            if user_ts is not None:
                logger.info(f"Latency multimodal user_transcription→bot_audio={now - user_ts:.3f}s")
        await self.push_frame(frame, direction)


def resolve_stt_language(language: str) -> Language:
    normalized = (language or "").strip().lower()
    if normalized.startswith("ar"):
        return Language.AR
    if normalized.startswith("en"):
        return Language.EN
    return Language.AR


def normalize_gemini_live_model_name(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return "models/gemini-2.0-flash-live-001"
    if model.startswith("models/"):
        return model
    if "/" in model:
        return model
    return f"models/{model}"


def build_multimodal_opening_message(greeting_text: str) -> str:
    greeting_text = (greeting_text or "").strip()
    return (
        "ابدأ المكالمة هلّق. أول جملة تحكيها لازم تكون EXACTLY هالنص بدون أي تغيير:\n"
        f"\"{greeting_text}\"\n"
        "بعدها اسأل سؤال واحد قصير للتأكيد وبس."
    )


def build_deepgram_live_options(*, encoding: str, sample_rate: int, language: str) -> LiveOptions:
    language = (language or "ar").strip()
    model = (os.getenv("DEEPGRAM_MODEL") or "").strip() or None
    if not model:
        model = "nova-2-phonecall" if language.startswith("en") else "nova-2"

    utterance_end_ms = "1000"
    utterance_end_ms_env = (os.getenv("DEEPGRAM_UTTERANCE_END_MS") or "").strip()
    if utterance_end_ms_env:
        try:
            utterance_end_ms = str(int(float(utterance_end_ms_env)))
        except Exception:
            utterance_end_ms = "1000"

    return LiveOptions(
        encoding=encoding,
        language=language,
        model=model,
        channels=1,
        sample_rate=sample_rate,
        interim_results=True,
        smart_format=True,
        punctuate=False,
        profanity_filter=False,
        vad_events=False,
        utterance_end_ms=utterance_end_ms,
    )


def normalize_jordanian_text(text: str) -> str:
    if not text:
        return text
    replacements = [
        ("لماذا", "ليش"),
        ("ماذا", "شو"),
        ("السائق", "الشوفير"),
        ("الآن", "هسا"),
        ("سوف", "رح"),
        ("تريد", "بدك"),
        ("أريد", "بدي"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def should_drop_tts_text(text: str) -> bool:
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    return re.fullmatch(r"[\.\,\!\?\u061F\u060C\u060D\u061B\u066A-\u066C\u06D4]+", stripped) is not None


class JordanianTTSPreprocessor(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            text = getattr(frame, "text", None)
            if isinstance(text, str):
                text = normalize_jordanian_text(text)
                if should_drop_tts_text(text):
                    return
                frame.text = text
        await self.push_frame(frame, direction)


class AudioSampleRateResampler(FrameProcessor):
    def __init__(self, *, target_input_sample_rate: int, target_output_sample_rate: int):
        super().__init__()
        self._target_in = int(target_input_sample_rate)
        self._target_out = int(target_output_sample_rate)
        self._in_resampler = create_stream_resampler()
        self._out_resampler = create_stream_resampler()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and frame.sample_rate != self._target_in:
            resampled = await self._in_resampler.resample(frame.audio, frame.sample_rate, self._target_in)
            frame = InputAudioRawFrame(audio=resampled, sample_rate=self._target_in, num_channels=frame.num_channels)
        elif isinstance(frame, OutputAudioRawFrame) and frame.sample_rate != self._target_out:
            resampled = await self._out_resampler.resample(frame.audio, frame.sample_rate, self._target_out)
            frame = OutputAudioRawFrame(audio=resampled, sample_rate=self._target_out, num_channels=frame.num_channels)
        await self.push_frame(frame, direction)


async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id:
        return
    telnyx_key = os.getenv("TELNYX_API_KEY")
    if not telnyx_key:
        return
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    encoded_call_control_id = quote(call_control_id, safe="")
    url = f"https://api.telnyx.com/v2/calls/{encoded_call_control_id}/actions/hangup"
    headers = {"Authorization": f"Bearer {telnyx_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"reason": "normal_clearing"}) as resp:
                if resp.status >= 300:
                    await resp.text()
    except Exception:
        return

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
    use_multimodal_live = os.getenv("USE_MULTIMODAL_LIVE", "false").lower() == "true"

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

    customer_name = lead_data.get("customer_name") or "العميل"
    order_items = lead_data.get("order_items") or ""
    delivery_time = lead_data.get("delivery_time") or ""
    greeting_text = (
        f"السلام عليكم، معك خالد من شركة كوكب للتوصيل. كيفك يا {customer_name}؟ "
        f"في إلك معنا طلب {order_items} المفروض يوصلك على الساعة {delivery_time}. "
        f"بس حبيت أتأكد إذا الأمور تمام ونبعتلك السائق؟"
    )
    
    use_smart_turn = (os.getenv("USE_SMART_TURN") or "true").strip().lower() in {"1", "true", "yes", "y"}
    vad_stop_secs = 0.2 if use_smart_turn else 0.4
    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.0, start_secs=0.2, stop_secs=vad_stop_secs, confidence=0.45))

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding=inbound_encoding,
        inbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(sample_rate=stream_sample_rate),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            session_timeout=300,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=stream_sample_rate,
            audio_out_sample_rate=stream_sample_rate,
        ),
    )

    system_prompt = f"""
# IDENTITY
You are "Khalid", a professional, polite, and native Jordanian delivery coordinator from "Kawkab Delivery" (شركة كوكب للتوصيل).
Your goal is to confirm delivery details with customers in a way that feels 100% human and local to Amman, Jordan.

# PERSONALITY & TONE
- **Voice**: Friendly, energetic, and "Ibn Nas" (well-mannered).
- **Style**: Use the Ammani/Urban Jordanian dialect.
- **Efficiency**: Jordanians appreciate quick calls. Keep responses under 10 words unless explaining something.
- **Rules**: Never speak Formal Arabic (Fusha). Never speak English unless the customer starts in English.
- **Jordanian Only**: If you catch yourself using non-Jordanian words, immediately rephrase in Jordanian.

# OPENING
- The opening line is already delivered by the system TTS. Do NOT repeat it.

# DIALECT GUIDELINES
- Use "G" for "Qaf" (e.g., 'Galleh' for 'Qalleh').
- Use local fillers: "يا هلا والله", "أبشر", "من عيوني", "عشان هيك", "هسا", "مية مية".
- Prefer Jordanian words: "شو" بدل "ماذا", "ليش" بدل "لماذا", "هسا" بدل "الآن", "بدك" بدل "تريد".
- If they say "Yes/Okay": Respond with "ممتاز، أبشر" or "مية مية، هسا برتب مع الشوفير".
- If they say "No/Cancel": Respond with "ولا يهمك، حصل خير. بس بقدر أعرف شو السبب للإلغاء؟".

# LOGIC & TOOLS
1. **Confirmation**: If they agree, call `update_lead_status_confirmed` and then say: "تمام، هسا بنرتب الأمور ويوصلك على الموعد إن شاء الله. مع السلامة."
2. **Cancellation**: If they cancel, call `update_lead_status_cancelled` and then say: "تم، لغينا الطلب. يومك سعيد، مع السلامة."
3. **Handling Interruption**: If the user says "Hello?" or "Are you there?" (ألو / معك؟ / وينك؟ / سلام عليكم / السلام عليكم), respond immediately with: "معك معك، تفضل..." and do NOT repeat the full introduction.
4. **Tool Safety**: Never confirm or cancel based on greetings or unclear words. Ask 1 short question if unclear.

# CONTEXT
- Customer: {lead_data['customer_name']}
- Items: {lead_data['order_items']}
- Time: {lead_data['delivery_time']}
"""

    if use_multimodal_live:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("USE_MULTIMODAL_LIVE=true but GOOGLE_API_KEY/GEMINI_API_KEY is missing. Falling back.")
        else:
            gemini_in_sample_rate = 16000
            try:
                gemini_in_sample_rate = int(os.getenv("GEMINI_LIVE_SAMPLE_RATE") or 16000)
            except Exception:
                gemini_in_sample_rate = 16000

            model = normalize_gemini_live_model_name(os.getenv("GEMINI_LIVE_MODEL") or "gemini-2.0-flash-live-001")
            voice_id = (os.getenv("GEMINI_LIVE_VOICE") or "Charon").strip()
            from pipecat.services.google.gemini_live.llm import (
                GeminiLiveLLMService as GeminiLiveService,
                InputParams as GeminiLiveInputParams,
            )

            gemini_params = GeminiLiveInputParams(temperature=0.3)
            try:
                gemini_params.language = Language.AR
            except Exception:
                pass
            try:
                gemini_params.sample_rate = gemini_in_sample_rate
            except Exception:
                pass
            try:
                gemini_params.mime_type = "audio/pcm"
            except Exception:
                pass
            logger.info(
                f"GeminiLive mode enabled (model={model}, voice={voice_id}, in_sr={gemini_in_sample_rate}, out_sr={stream_sample_rate})"
            )
            try:
                gemini_live = GeminiLiveService(
                    api_key=api_key,
                    model=model,
                    voice_id=voice_id,
                    system_instruction=system_prompt,
                    params=gemini_params,
                )
            except Exception as e:
                logger.error(f"GeminiLive init failed; falling back. ({e})")
                gemini_live = None

            call_end_delay_s = 2.5
            try:
                call_end_delay_s = float(os.getenv("CALL_END_DELAY_S") or 2.5)
            except Exception:
                call_end_delay_s = 2.5

            async def confirm_order(params: FunctionCallParams):
                logger.info(f"Confirming order for lead {lead_data['id']}")
                reason = None
                try:
                    reason = params.arguments.get("reason")
                except Exception:
                    reason = None
                update_lead_status(lead_data["id"], "CONFIRMED")
                await params.result_callback({"value": "Order confirmed successfully.", "reason": reason})
                if call_control_id:
                    asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

            async def cancel_order(params: FunctionCallParams):
                logger.info(f"Cancelling order for lead {lead_data['id']}")
                reason = None
                try:
                    reason = params.arguments.get("reason")
                except Exception:
                    reason = None
                update_lead_status(lead_data["id"], "CANCELLED")
                await params.result_callback({"value": "Order cancelled.", "reason": reason})
                if call_control_id:
                    asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

            if gemini_live is not None:
                gemini_live.register_function("update_lead_status_confirmed", confirm_order)
                gemini_live.register_function("update_lead_status_cancelled", cancel_order)

            resampler = AudioSampleRateResampler(
                target_input_sample_rate=gemini_in_sample_rate,
                target_output_sample_rate=stream_sample_rate,
            )
            mm_perf = MultimodalPerf()
            if gemini_live is not None:
                pipeline = Pipeline([transport.input(), resampler, gemini_live, mm_perf, resampler, transport.output()])
                task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
                runner = PipelineRunner()

                async def multimodal_stuck_watchdog():
                    timeout_s = 4.0
                    try:
                        timeout_s = float(os.getenv("MULTIMODAL_STUCK_TIMEOUT_S") or 4.0)
                    except Exception:
                        timeout_s = 4.0
                    while True:
                        await asyncio.sleep(0.5)
                        user_ts = _MM.get("last_user_transcription_ts")
                        bot_ts = _MM.get("last_bot_started_ts")
                        if user_ts is None:
                            continue
                        if bot_ts is not None and bot_ts > user_ts:
                            continue
                        if time.monotonic() - user_ts >= timeout_s:
                            _MM["last_user_transcription_ts"] = None
                            await task.queue_frames(
                                [
                                    LLMMessagesAppendFrame(
                                        [{"role": "user", "content": "رد بسرعة وباختصار."}],
                                        run_llm=True,
                                    )
                                ]
                            )

                asyncio.create_task(multimodal_stuck_watchdog())
                opening_message = build_multimodal_opening_message(greeting_text)
                logger.info("Queuing multimodal opening message + LLMRunFrame")
                await task.queue_frames(
                    [
                        LLMMessagesAppendFrame([{"role": "user", "content": opening_message}], run_llm=False),
                        LLMRunFrame(),
                    ]
                )
                await runner.run(task)
                return

    stt_language = os.getenv("STT_LANGUAGE") or os.getenv("DEEPGRAM_LANGUAGE") or "ar"
    stt_language_enum = resolve_stt_language(stt_language)
    stt_provider = (os.getenv("STT_PROVIDER") or "").strip().lower()
    if not stt_provider:
        stt_provider = "google" if stt_language_enum == Language.AR else "deepgram"

    if stt_provider == "deepgram" and stt_language_enum == Language.AR:
        logger.warning("Deepgram does not support Arabic streaming. Switching STT_PROVIDER to google.")
        stt_provider = "google"

    if stt_provider == "google":
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
    else:
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_key:
            logger.error("CRITICAL: DEEPGRAM_API_KEY is missing in environment variables!")
        else:
            logger.info(f"Deepgram API Key found (len={len(deepgram_key)}).")
        dg_encoding = "mulaw" if inbound_encoding == "PCMU" else "alaw" if inbound_encoding == "PCMA" else "linear16"
        logger.info(f"Initializing Deepgram with encoding: {dg_encoding} (inbound: {inbound_encoding})")
        stt = DeepgramSTTService(
            api_key=deepgram_key,
            model=os.getenv("DEEPGRAM_MODEL") or "nova-2-phonecall",
            language=os.getenv("DEEPGRAM_LANGUAGE") or "en",
            sample_rate=stream_sample_rate,
            encoding=dg_encoding,
        )
    
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
    fallback_voice_id = "ar-XA-Chirp3-HD-Charon"
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
            logger.warning(f"Google TTS voice not found: {voice_id}. Falling back to {fallback_voice_id}.")
            voice_id = fallback_voice_id
    except Exception as e:
        if voice_id.startswith("ar-JO"):
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
    call_end_delay_s = 2.5
    try:
        call_end_delay_s = float(os.getenv("CALL_END_DELAY_S") or 2.5)
    except Exception:
        call_end_delay_s = 2.5

    async def confirm_order(params: FunctionCallParams):
        logger.info(f"Confirming order for lead {lead_data['id']}")
        reason = None
        try:
            reason = params.arguments.get("reason")
        except Exception:
            reason = None
        update_lead_status(lead_data['id'], 'CONFIRMED')
        await params.result_callback({"value": "Order confirmed successfully.", "reason": reason})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

    async def cancel_order(params: FunctionCallParams):
        logger.info(f"Cancelling order for lead {lead_data['id']}")
        reason = None
        try:
            reason = params.arguments.get("reason")
        except Exception:
            reason = None
        update_lead_status(lead_data['id'], 'CANCELLED')
        await params.result_callback({"value": "Order cancelled.", "reason": reason})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

    llm.register_function(
        "update_lead_status_confirmed",
        confirm_order
    )
    llm.register_function(
        "update_lead_status_cancelled",
        cancel_order
    )

    messages = [{"role": "system", "content": system_prompt}]
    context = LLMContext(messages=messages)

    start_strategies = []
    stop_strategies = []
    if use_smart_turn:
        try:
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
            from pipecat.turns.user_start import VADUserTurnStartStrategy
            from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

            start_strategies = [VADUserTurnStartStrategy(enable_interruptions=True)]
            stop_strategies = [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            logger.info("Smart Turn enabled (LocalSmartTurnAnalyzerV3).")
        except Exception as e:
            logger.warning(f"Smart Turn unavailable; falling back to transcription turn stop. ({e})")
            use_smart_turn = False
    if not use_smart_turn:
        start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
        stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=0.4)]

    user_turn_strategies = UserTurnStrategies(start=start_strategies, stop=stop_strategies)

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=user_turn_strategies,
            user_mute_strategies=[],
            user_turn_stop_timeout=0.5,
        ),
    )

    stt_perf = STTPerf()
    llm_perf = LLMPerf()
    tts_perf = TTSPerf()

    pipeline = Pipeline([
        transport.input(),
        stt,
        stt_perf,
        aggregators.user(),
        llm,
        llm_perf,
        JordanianTTSPreprocessor(),
        tts,
        tts_perf,
        transport.output(),
        aggregators.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    logger.info(f"Queuing greeting (inbound_encoding={inbound_encoding}, tts_encoding={tts_encoding})")
    await task.queue_frames([TTSSpeakFrame(greeting_text)])

    async def post_greeting_follow_up():
        await asyncio.sleep(7.0)
        if _PERF.get("last_user_text_ts") is None:
            await task.queue_frames([TTSSpeakFrame("تمام؟ بتسمعني؟")])

    asyncio.create_task(post_greeting_follow_up())
    logger.info("Starting single pipeline run")
    await runner.run(task)
