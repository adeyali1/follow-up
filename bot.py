import os
import asyncio
import aiohttp
from urllib.parse import quote
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMRunFrame
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy
from services.supabase_service import update_lead_status

import json
import time
from pipecat.audio.utils import create_stream_resampler

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None}
BOT_BUILD_ID = "2026-01-20-multimodal-llmrunframe"


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


class MultimodalTranscriptRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float):
        super().__init__()
        self._delay_s = float(delay_s)
        self._queue_frames = None
        self._pending = None
        self._last_user_transcript_ts = None

    def set_queue_frames(self, queue_frames):
        self._queue_frames = queue_frames

    async def _schedule(self, ts: float):
        try:
            await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError:
            return
        if self._queue_frames is None:
            return
        if self._last_user_transcript_ts != ts:
            return
        logger.info("Multimodal: user transcript idle → LLMRunFrame")
        await self._queue_frames([LLMRunFrame()])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            is_final = True
            try:
                is_final = bool(getattr(frame, "is_final", True))
            except Exception:
                is_final = True
            if not is_final:
                await self.push_frame(frame, direction)
                return
            now = time.monotonic()
            try:
                text = (getattr(frame, "text", None) or "").strip()
                if text:
                    logger.debug(f"Multimodal: final user transcript received ({len(text)} chars)")
            except Exception:
                pass
            self._last_user_transcript_ts = now
            if self._pending is not None:
                self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)


class LeadStatusTranscriptFallback(FrameProcessor):
    def __init__(self, *, lead_id: str, call_control_id: str | None, call_end_delay_s: float, finalized_ref: dict):
        super().__init__()
        self._lead_id = lead_id
        self._call_control_id = call_control_id
        self._call_end_delay_s = float(call_end_delay_s)
        self._finalized_ref = finalized_ref

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").strip().lower()
        for ch in ["\n", "\r", "\t", ".", ",", "!", "?", "؟", "،", "؛", "\"", "'"]:
            text = text.replace(ch, " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text

    @staticmethod
    def _is_confirm(text: str) -> bool:
        t = LeadStatusTranscriptFallback._normalize(text)
        if not t:
            return False
        if any(x in t for x in ["الغ", "كنسل", "cancel", "مش بد", "مش بدي", "لا بدي", "إلغاء", "الغاء"]):
            return False
        if "مش" in t and any(x in t for x in ["تمام", "ماشي", "موافق"]):
            return False
        return any(
            x in t
            for x in [
                "تمام",
                "ماشي",
                "أكيد",
                "اكيد",
                "موافق",
                "اوكي",
                "okay",
                "ok",
                "yes",
                "بنعم",
                "اه",
                "أه",
            ]
        )

    @staticmethod
    def _is_cancel(text: str) -> bool:
        t = LeadStatusTranscriptFallback._normalize(text)
        if not t:
            return False
        return any(x in t for x in ["الغ", "إلغاء", "الغاء", "كنسل", "cancel", "مش بد", "مش بدي"])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            try:
                if self._finalized_ref.get("value"):
                    await self.push_frame(frame, direction)
                    return
            except Exception:
                pass
            is_final = True
            try:
                is_final = bool(getattr(frame, "is_final", True))
            except Exception:
                is_final = True
            if is_final:
                text = ""
                try:
                    text = str(getattr(frame, "text", "") or "")
                except Exception:
                    text = ""
                if self._is_confirm(text):
                    self._finalized_ref["value"] = "CONFIRMED"
                    logger.info("Fallback: detected confirmation from transcript; updating lead status CONFIRMED")
                    update_lead_status(self._lead_id, "CONFIRMED")
                    if self._call_control_id:
                        asyncio.create_task(hangup_telnyx_call(self._call_control_id, self._call_end_delay_s))
                elif self._is_cancel(text):
                    self._finalized_ref["value"] = "CANCELLED"
                    logger.info("Fallback: detected cancellation from transcript; updating lead status CANCELLED")
                    update_lead_status(self._lead_id, "CANCELLED")
                    if self._call_control_id:
                        asyncio.create_task(hangup_telnyx_call(self._call_control_id, self._call_end_delay_s))
        await self.push_frame(frame, direction)


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
    return greeting_text


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

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot for lead: {lead_data['id']}")
    logger.info(f"Bot build: {BOT_BUILD_ID}")
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

    customer_name = lead_data.get("customer_name") or "العميل"
    greeting_text = f"السلام عليكم، معك خالد من شركة كوكب للتوصيل. معك يا {customer_name}؟"
    vad_stop_secs = 0.2
    try:
        vad_stop_secs = float(os.getenv("VAD_STOP_SECS") or 0.2)
    except Exception:
        vad_stop_secs = 0.2
    try:
        vad_min_volume = float(os.getenv("VAD_MIN_VOLUME") or 0.6)
    except Exception:
        vad_min_volume = 0.6
    try:
        vad_confidence = float(os.getenv("VAD_CONFIDENCE") or 0.7)
    except Exception:
        vad_confidence = 0.7
    vad = SileroVADAnalyzer(
        params=VADParams(min_volume=vad_min_volume, start_secs=0.2, stop_secs=vad_stop_secs, confidence=vad_confidence)
    )

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
# ROLE
You are Khalid, a real-life, professional delivery coordinator for \"Kawkab Delivery\" in Amman, Jordan. This is a live phone conversation. Your goal is to be helpful, crisp, and sound 100% like a polite Jordanian man.

# VOCAL PERSONALITY & TONE (CRITICAL FOR LIVE API)
- Voice Tone: Warm, energetic, and professional.
- Pacing: Speak at a natural human speed. Do not rush, but do not drone.
- Vocalisms: Use natural Jordanian fillers like \"تمام\", \"يا هلا\", \"طيب\" to show you are listening.
- Emotion: If the customer is happy, be warm. If they are in a hurry, be faster and more efficient.

# LANGUAGE & DIALECT (WHITE JORDANIAN)
- Primary Dialect: Educated Ammani (White Arabic).
- Phonetics: Use the soft 'G' for 'Qaf' in casual words, but keep the clear 'Q' for professional words like \"تأكيد\".
- Forbidden Words: NEVER use Fusha words like \"ماذا\", \"هل ترغب\", \"سوف\", or \"السائق\".
- Preferred Words: Use \"بدي\", \"المندوب\", \"عشان\", \"هسا\".
- Local Courtesy: Use \"غلبناك معنا\" and \"على راسي\" to build trust.

# LIVE CONVERSATION LOGIC
- Turn-Taking: If the user interrupts you, STOP talking immediately and listen.
- Greeting State: If the user says \"Hello\" or \"Salam\" first, do NOT repeat your full intro. Just say: \"يا هلا والله، معك خالد... بس كنت حاب أتأكد من طلبك...\".
- Silence Handling: If the user is silent for too long, ask politely: \"معي يا غالي؟\" or \"ألو؟ عدي معي؟\".
- Brevity: Phone calls are expensive and users are busy. Keep 90% of your responses under 10 words.

# GREETING RULE (IMPORTANT)
- Your very first spoken line must match the prepared greeting message in the conversation context (assistant message). Say it once, then wait for the customer.

# TASK WORKFLOW
1. Confirm: \"عدي، طلبك {lead_data['order_items']} رح يوصل ع الساعة {lead_data['delivery_time']}. بنعتمد؟\"
2. Success: If confirmed, call update_lead_status_confirmed immediately, then say: \"مية مية، هسا برتب مع المندوب ويوصلك ع الموعد. غلبناك!\"
3. Cancellation: If they cancel, call update_lead_status_cancelled immediately, then say: \"ولا يهمك، حصل خير. لغينا الطلب وبنتمنى نخدمك مرة تانية.\"

# CONTEXT
- Customer: {lead_data['customer_name']}
- Items: {lead_data['order_items']}
- Time: {lead_data['delivery_time']}
"""

    if use_multimodal_live:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("USE_MULTIMODAL_LIVE=true but GOOGLE_API_KEY/GEMINI_API_KEY is missing.")
            return

        gemini_in_sample_rate = 16000
        try:
            gemini_in_sample_rate = int(os.getenv("GEMINI_LIVE_SAMPLE_RATE") or 16000)
        except Exception:
            gemini_in_sample_rate = 16000

        model_env = (os.getenv("GEMINI_LIVE_MODEL") or "").strip()
        model = normalize_gemini_live_model_name(model_env) if model_env else None
        voice_id = (os.getenv("GEMINI_LIVE_VOICE") or "Charon").strip()
        from pipecat.services.google.gemini_live.llm import (
            GeminiLiveLLMService as GeminiLiveService,
            InputParams as GeminiLiveInputParams,
        )
        try:
            from pipecat.services.google.gemini_live.llm import GeminiModalities
        except Exception:
            GeminiModalities = None

        gemini_params = GeminiLiveInputParams(temperature=0.3)
        gemini_language_env = (os.getenv("GEMINI_LIVE_LANGUAGE") or "").strip()
        if gemini_language_env:
            try:
                if gemini_language_env.lower().startswith("en"):
                    gemini_params.language = Language.EN
                elif gemini_language_env.lower().startswith("ar"):
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
        try:
            if GeminiModalities is not None:
                gemini_params.modalities = GeminiModalities.AUDIO
        except Exception:
            pass
        modality_label = "DEFAULT"
        try:
            modality_label = str(getattr(gemini_params, "modalities", None) or "DEFAULT")
        except Exception:
            modality_label = "DEFAULT"
        logger.info(
            f"GeminiLive mode enabled (model={model or 'DEFAULT'}, voice={voice_id}, modalities={modality_label}, in_sr={gemini_in_sample_rate}, out_sr={stream_sample_rate})"
        )
        try:
            gemini_kwargs = {
                "api_key": api_key,
                "voice_id": voice_id,
                "system_instruction": system_prompt,
                "params": gemini_params,
                "inference_on_context_initialization": False,
            }
            if model:
                gemini_kwargs["model"] = model
            gemini_live = GeminiLiveService(**gemini_kwargs)
        except Exception as e:
            logger.error(f"GeminiLive init failed. ({e})")
            return

        call_end_delay_s = 2.5
        try:
            call_end_delay_s = float(os.getenv("CALL_END_DELAY_S") or 2.5)
        except Exception:
            call_end_delay_s = 2.5
        lead_finalized = {"value": None}

        async def confirm_order(params: FunctionCallParams):
            logger.info(f"Confirming order for lead {lead_data['id']}")
            lead_finalized["value"] = "CONFIRMED"
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
            lead_finalized["value"] = "CANCELLED"
            reason = None
            try:
                reason = params.arguments.get("reason")
            except Exception:
                reason = None
            update_lead_status(lead_data["id"], "CANCELLED")
            await params.result_callback({"value": "Order cancelled.", "reason": reason})
            if call_control_id:
                asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

        gemini_live.register_function("update_lead_status_confirmed", confirm_order)
        gemini_live.register_function("update_lead_status_cancelled", cancel_order)

        input_resampler = AudioSampleRateResampler(
            target_input_sample_rate=stream_sample_rate,
            target_output_sample_rate=gemini_in_sample_rate,
        )
        output_resampler = AudioSampleRateResampler(
            target_input_sample_rate=gemini_in_sample_rate,
            target_output_sample_rate=stream_sample_rate,
        )
        mm_perf = MultimodalPerf()
        logger.info(f"Multimodal resamplers: in {stream_sample_rate}->{gemini_in_sample_rate}, out {gemini_in_sample_rate}->{stream_sample_rate}")

        mm_context = LLMContext(messages=[{"role": "assistant", "content": greeting_text}])
        user_mute_strategies = []
        mute_first_bot = (os.getenv("MULTIMODAL_MUTE_UNTIL_FIRST_BOT") or "false").lower() == "true"
        if mute_first_bot:
            try:
                from pipecat.turns.mute import MuteUntilFirstBotCompleteUserMuteStrategy

                user_mute_strategies.append(MuteUntilFirstBotCompleteUserMuteStrategy())
            except Exception:
                pass
        logger.info(f"Multimodal mute until first bot complete: {bool(user_mute_strategies)}")
        stop_timeout_s = 0.8
        try:
            stop_timeout_s = float(os.getenv("MULTIMODAL_TURN_STOP_TIMEOUT_S") or 0.8)
        except Exception:
            stop_timeout_s = 0.8
        start_strategies = []
        try:
            from pipecat.turns.user_start import VADUserTurnStartStrategy

            start_strategies = [VADUserTurnStartStrategy(enable_interruptions=False)]
        except Exception:
            start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
        stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=stop_timeout_s)]
        mm_aggregators = LLMContextAggregatorPair(
            mm_context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies),
                user_mute_strategies=user_mute_strategies,
            ),
        )
        try:
            maybe_coro = gemini_live.set_context(mm_context)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        except Exception:
            pass

        @gemini_live.event_handler("on_error")
        async def _on_gemini_live_error(service, error):
            msg = str(error)
            if "Unsupported language code 'ar-XA'" in msg or "Unsupported language code" in msg:
                live_connection_failed["value"] = True
                logger.error(
                    "GeminiLive language mismatch. Remove GEMINI_LIVE_LANGUAGE or set GEMINI_LIVE_LANGUAGE=en-US."
                )
                logger.error(msg)
            elif "received 1008" in msg or "policy violation" in msg or "is not found" in msg or "bidiGenerateContent" in msg:
                live_connection_failed["value"] = True
                logger.error(f"GeminiLive rejected model/key: {msg}")
            else:
                logger.error(f"GeminiLive error: {msg}")

        transcript_run_delay_s = 0.7
        try:
            transcript_run_delay_s = float(os.getenv("MULTIMODAL_TRANSCRIPT_STOP_S") or 0.7)
        except Exception:
            transcript_run_delay_s = 0.7
        transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=transcript_run_delay_s)
        transcript_fallback = LeadStatusTranscriptFallback(
            lead_id=lead_data["id"],
            call_control_id=call_control_id,
            call_end_delay_s=call_end_delay_s,
            finalized_ref=lead_finalized,
        )

        pipeline = Pipeline(
            [
                transport.input(),
                mm_aggregators.user(),
                input_resampler,
                gemini_live,
                transcript_trigger,
                transcript_fallback,
                mm_perf,
                output_resampler,
                transport.output(),
                mm_aggregators.assistant(),
            ]
        )
        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
        transcript_trigger.set_queue_frames(task.queue_frames)
        runner = PipelineRunner()
        did_trigger_initial_run = {"value": False}
        live_connection_failed = {"value": False}

        @transport.event_handler("on_client_connected")
        async def _on_client_connected(_transport, _client):
            if did_trigger_initial_run["value"]:
                return
            did_trigger_initial_run["value"] = True
            logger.info("Multimodal: client connected, triggering initial LLMRunFrame")
            await task.queue_frames([LLMRunFrame()])

        async def multimodal_first_turn_failsafe():
            timeout_s = 8.0
            try:
                timeout_s = float(os.getenv("MULTIMODAL_FIRST_TURN_FAILSAFE_S") or 8.0)
            except Exception:
                timeout_s = 8.0
            await asyncio.sleep(timeout_s)
            if _MM.get("last_bot_started_ts") is None and not live_connection_failed["value"]:
                logger.warning("Multimodal: first bot audio not detected yet; re-triggering LLMRunFrame")
                await task.queue_frames([LLMRunFrame()])

        async def multimodal_stuck_watchdog():
            timeout_s = 10.0
            try:
                timeout_s = float(os.getenv("MULTIMODAL_STUCK_TIMEOUT_S") or 10.0)
            except Exception:
                timeout_s = 10.0
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
                            LLMMessagesAppendFrame([{"role": "user", "content": "رد بسرعة وباختصار."}], run_llm=False),
                            LLMRunFrame(),
                        ]
                    )

        async def multimodal_silent_start_retry():
            timeout_s = 6.0
            try:
                timeout_s = float(os.getenv("MULTIMODAL_START_TIMEOUT_S") or 6.0)
            except Exception:
                timeout_s = 6.0
            max_retries = 3
            try:
                max_retries = int(os.getenv("MULTIMODAL_MAX_START_RETRIES") or 3)
            except Exception:
                max_retries = 3
            for attempt in range(max_retries):
                await asyncio.sleep(timeout_s)
                if live_connection_failed["value"]:
                    logger.error("Multimodal: live connection failed (model/key). Stop retrying.")
                    return
                if _MM.get("last_bot_started_ts") is not None:
                    return
                logger.warning(f"Multimodal: no bot audio detected, retrying LLMRunFrame (attempt {attempt + 1}/{max_retries})")
                await task.queue_frames(
                    [
                        LLMMessagesAppendFrame([{"role": "user", "content": "احكي هسا بصوت واضح."}], run_llm=False),
                        LLMRunFrame(),
                    ]
                )

        if (os.getenv("MULTIMODAL_ENABLE_START_RETRY") or "false").lower() == "true":
            asyncio.create_task(multimodal_silent_start_retry())
        if (os.getenv("MULTIMODAL_ENABLE_FIRST_TURN_FAILSAFE") or "false").lower() == "true":
            asyncio.create_task(multimodal_first_turn_failsafe())
        if (os.getenv("MULTIMODAL_ENABLE_STUCK_WATCHDOG") or "true").lower() == "true":
            asyncio.create_task(multimodal_stuck_watchdog())
        await runner.run(task)
        return
    logger.error("Classic STT/Vertex/TTS pipeline has been removed. Set USE_MULTIMODAL_LIVE=true.")
    return
