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
from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy

# --- MOCK SERVICE FOR STANDALONE TESTING ---
# If you have your own file, uncomment your import and remove this function
# from services.supabase_service import update_lead_status
def update_lead_status(lead_id, status):
    logger.info(f"--- DATABASE UPDATE: LEAD {lead_id} STATUS -> {status} ---")

import json
import time

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-20-dental-coordinator-v1"
_VAD_MODEL = {"value": None}


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
        _MM["last_llm_run_ts"] = time.monotonic()
        await self._queue_frames([LLMRunFrame()])

    def cancel_pending(self):
        if self._pending is not None:
            try:
                self._pending.cancel()
            except Exception:
                pass
            self._pending = None

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


class MultimodalUserStopRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float = 0.05, min_interval_s: float = 0.25):
        super().__init__()
        self._delay_s = float(delay_s)
        self._min_interval_s = float(min_interval_s)
        self._queue_frames = None
        self._pending = None
        self._last_stop_ts = None

    def set_queue_frames(self, queue_frames):
        self._queue_frames = queue_frames

    def cancel_pending(self):
        if self._pending is not None:
            try:
                self._pending.cancel()
            except Exception:
                pass
            self._pending = None

    async def _schedule(self, ts: float):
        try:
            await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError:
            return
        if self._queue_frames is None:
            return
        if self._last_stop_ts != ts:
            return
        last_run = _MM.get("last_llm_run_ts")
        now = time.monotonic()
        if last_run is not None and (now - last_run) < self._min_interval_s:
            return
        _MM["last_llm_run_ts"] = now
        logger.info("Multimodal: user stop → LLMRunFrame")
        await self._queue_frames([LLMRunFrame()])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if type(frame).__name__ == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            self._last_stop_ts = now
            if self._pending is not None:
                self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)


class TurnStateLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._user_turn_started_ts = None
        self._logged_bot_audio = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "UserStartedSpeakingFrame":
            self._user_turn_started_ts = time.monotonic()
            logger.info("Turn: user started speaking")
        elif frame_name == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            dur = None
            if self._user_turn_started_ts is not None:
                dur = now - self._user_turn_started_ts
            self._user_turn_started_ts = None
            if dur is None:
                logger.info("Turn: user stopped speaking")
            else:
                logger.info(f"Turn: user stopped speaking (dur={dur:.2f}s)")
        elif frame_name in {"BotStartedSpeakingFrame", "TTSStartedFrame"} and not self._logged_bot_audio:
            self._logged_bot_audio = True
            logger.info(f"Turn: first bot audio started ({frame_name})")
        await self.push_frame(frame, direction)


class OutboundAudioLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._logged = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not self._logged and isinstance(frame, AudioRawFrame):
            self._logged = True
            try:
                logger.info(f"AudioOut: first audio frame sr={frame.sample_rate} bytes={len(frame.audio)}")
            except Exception:
                logger.info("AudioOut: first audio frame")
        await self.push_frame(frame, direction)


class InboundAudioLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._logged = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not self._logged and isinstance(frame, InputAudioRawFrame):
            self._logged = True
            try:
                logger.info(f"AudioInDecoded: first audio frame sr={frame.sample_rate} bytes={len(frame.audio)}")
            except Exception:
                logger.info("AudioInDecoded: first audio frame")
        await self.push_frame(frame, direction)


class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 40):
        super().__init__()
        self._chunk_ms = int(chunk_ms)
        self._pace = (os.getenv("AUDIO_OUT_PACE") or "true").lower() == "true"

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self._chunk_ms <= 0:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, AudioRawFrame):
            audio = frame.audio
            if isinstance(audio, (bytes, bytearray)) and len(audio) > 0:
                sample_rate = int(getattr(frame, "sample_rate", 0) or 0)
                channels = int(getattr(frame, "num_channels", 1) or 1)
                if sample_rate > 0 and channels > 0 and self._chunk_ms > 0:
                    bytes_per_sample = 2
                    bytes_per_frame = channels * bytes_per_sample
                    chunk_bytes = int(sample_rate * self._chunk_ms / 1000) * bytes_per_frame
                    chunk_bytes = max(chunk_bytes - (chunk_bytes % bytes_per_frame), bytes_per_frame)
                    if len(audio) > chunk_bytes:
                        frame_type = type(frame)
                        first = True
                        for i in range(0, len(audio), chunk_bytes):
                            chunk = audio[i : i + chunk_bytes]
                            if not chunk:
                                continue
                            try:
                                await self.push_frame(
                                    frame_type(audio=chunk, sample_rate=frame.sample_rate, num_channels=frame.num_channels),
                                    direction,
                                )
                                if self._pace and not first:
                                    await asyncio.sleep(self._chunk_ms / 1000)
                            except Exception:
                                await self.push_frame(frame, direction)
                                return
                            first = False
                        return
        await self.push_frame(frame, direction)


class AppointmentStatusTranscriptFallback(FrameProcessor):
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
        t = AppointmentStatusTranscriptFallback._normalize(text)
        if not t:
            return False
        # Avoid false positives if they are cancelling
        if any(x in t for x in ["الغ", "كنسل", "cancel", "مش جاي", "ما بقدر", "صعبة", "لا"]):
            return False
        # Check for strong confirmation words in Ammani
        return any(
            x in t
            for x in [
                "تمام", "ماشي", "أكيد", "اكيد", "موافق", "ان شاء الله", "إن شاء الله",
                "جاي", "بكون موجود", "تم", "اوكي", "ok", "yes", "confirm", "100"
            ]
        )

    @staticmethod
    def _is_cancel(text: str) -> bool:
        t = AppointmentStatusTranscriptFallback._normalize(text)
        if not t:
            return False
        return any(x in t for x in ["الغ", "إلغاء", "الغاء", "كنسل", "cancel", "مش رح اقدر", "ما بقدر", "اجل", "أجل"])

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
                # Simple keyword matching as fallback to LLM function calling
                if self._is_confirm(text):
                    logger.info("Fallback: detected confirmation from transcript")
                    # We don't hard update status here, we let the LLM do it via function call usually,
                    # but if you want transcript enforcement, uncomment below:
                    # self._finalized_ref["value"] = "CONFIRMED"
                    # update_lead_status(self._lead_id, "CONFIRMED")
                elif self._is_cancel(text):
                    logger.info("Fallback: detected cancellation from transcript")
                    # self._finalized_ref["value"] = "CANCELLED"
                    # update_lead_status(self._lead_id, "CANCELLED")
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


def normalize_customer_name_for_ar(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return raw
    # Simple transliteration map if needed, or just return raw
    return raw


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
    logger.info(f"Starting DENTAL bot for lead: {lead_data['id']}")
    
    _MM["last_user_transcription_ts"] = None
    _MM["last_bot_started_ts"] = None
    _MM["last_llm_run_ts"] = None
    
    # Force multimodal for this professional demo
    use_multimodal_live = True 
    pipeline_sample_rate = 16000
    try:
        pipeline_sample_rate = int(os.getenv("PIPELINE_SAMPLE_RATE") or os.getenv("GEMINI_LIVE_SAMPLE_RATE") or 16000)
    except Exception:
        pipeline_sample_rate = 16000

    # ------------------------------------------------------------------
    # 0. Handle Telnyx Handshake to get stream_id
    # ------------------------------------------------------------------
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU"
    stream_sample_rate = 8000
    try:
        logger.info("Waiting for Telnyx 'start' event with stream_id...")
        for _ in range(3): 
            msg_text = await websocket_client.receive_text()
            logger.info(f"Received Telnyx message: {msg_text}")
            msg = json.loads(msg_text)
            
            if "media_format" in msg:
                 encoding = msg["media_format"].get("encoding", "").upper()
                 stream_sample_rate = int(msg["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                 if encoding == "PCMA":
                     inbound_encoding = "PCMA"
                 elif encoding == "PCMU":
                     inbound_encoding = "PCMU"

            elif "start" in msg and "media_format" in msg["start"]:
                 encoding = msg["start"]["media_format"].get("encoding", "").upper()
                 stream_sample_rate = int(msg["start"]["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                 if encoding == "PCMA":
                     inbound_encoding = "PCMA"
                 elif encoding == "PCMU":
                     inbound_encoding = "PCMU"

            if "stream_id" in msg:
                stream_id = msg["stream_id"]
                break
            elif "data" in msg and "stream_id" in msg["data"]:
                stream_id = msg["data"]["stream_id"]
                break
            elif msg.get("event") == "start":
                 if "stream_id" in msg:
                      stream_id = msg["stream_id"]
                      break
            
    except Exception as e:
        logger.error(f"Failed to capture stream_id from initial message: {e}")

    # ------------------------------------------------------------------
    # 1. Dental Coordinator Configuration
    # ------------------------------------------------------------------
    customer_name = normalize_customer_name_for_ar(lead_data.get("customer_name") or "عزيزي")
    # Default inputs if missing
    appointment_type = lead_data.get("appointment_type", "تنظيف وتبييض أسنان")
    appointment_time = lead_data.get("appointment_time", "بكرا الساعة 4 العصر")

    # Initial Greeting (Spoken first)
    greeting_text = f"مرحبا {customer_name}، معك سارة من عيادة إليت للأسنان. سامعني واضح؟"

    # VAD Settings
    vad_stop_secs = 0.4
    vad_min_volume = 0.6
    vad = None
    cached_vad = _VAD_MODEL.get("value")
    if cached_vad is None:
        vad = SileroVADAnalyzer(
            params=VADParams(min_volume=vad_min_volume, start_secs=0.2, stop_secs=vad_stop_secs, confidence=0.7)
        )
        _VAD_MODEL["value"] = vad
    else:
        vad = cached_vad

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding=inbound_encoding,
        inbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(sample_rate=pipeline_sample_rate),
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
            audio_in_sample_rate=pipeline_sample_rate,
            audio_out_sample_rate=pipeline_sample_rate,
        ),
    )

    # ------------------------------------------------------------------
    # 2. The Professional System Prompt
    # ------------------------------------------------------------------
    system_prompt = f"""
# ROLE
You are **Sara**, the Treatment Coordinator at **"Elite Dental Clinic"** (عيادة إليت للأسنان) in Amman, Jordan.
You are calling to confirm an upcoming appointment. This is a high-end, professional clinic.

# DIALECT (JORDANIAN - AMMANI)
- Speak educated, warm, professional Ammani Arabic.
- Use polite terms: "حضرتك", "أهلا وسهلا", "تمام", "يا هلا".
- Avoid slang or overly street language. Avoid classical Fusha (don't say "سوف" or "لماذا").
- Use English medical terms if natural (e.g., "Check-up", "Whitening" is okay if the user understands), but stick to Arabic mostly.

# CONTEXT
- Patient Name: {customer_name}
- Treatment: {appointment_type}
- Time: {appointment_time}
- Location: Abdoun, near the Embassy.
- Parking: Valet is available.

# WORKFLOW
1. **Greeting:** (Already spoken: "Hello {customer_name}, Sara from Elite Dental. Can you hear me?")
   - Wait for them to say "Yes" or "Ah".
2. **Confirmation:**
   - "حبيت أأكد موعدك {appointment_time} عشان {appointment_type}. بانتظارك دكتور أسامة."
   - (Translation: Wanted to confirm your appointment [time] for [type]. Dr. Osama is expecting you.)
3. **If Confirmed:**
   - Call function `confirm_appointment`.
   - Say: "ممتاز! يا ريت لو تيجوا قبل 10 دقايق عشان الإجراءات. بتشرفونا."
   - End politeley.
4. **If they want to Cancel/Reschedule:**
   - Show empathy. "سلامتك، ما في مشكلة."
   - Call function `cancel_appointment` (or ask when they want to move it).
   - Say: "ولا يهمك، بخلي قسم المواعيد يتواصل معك لترتيب وقت تاني. شكراً إلك."
5. **Handling Questions:**
   - **Price:** "الكشفية 20 دينار، والعلاج حسب الحالة الدكتور بحددلك."
   - **Pain:** "ما تخاف، الدكتور أسامة إيده خفيفة وبنستخدم تخدير موضعي ممتاز."
   - **Location:** "احنا بعبدون، قرب السفارة. وفي فاليت لسيارتك."

# RULES
- Keep responses short (1-2 sentences).
- Be extremely polite.
- Do NOT repeat the greeting if the conversation is flowing.
"""

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY is missing.")
        return

    # ------------------------------------------------------------------
    # 3. Gemini Live Setup
    # ------------------------------------------------------------------
    from pipecat.services.google.gemini_live.llm import (
        GeminiLiveLLMService as GeminiLiveService,
        InputParams as GeminiLiveInputParams,
    )
    
    # Optional: Use a specific model if needed
    model_env = os.getenv("GEMINI_LIVE_MODEL")
    model = normalize_gemini_live_model_name(model_env) if model_env else "models/gemini-2.0-flash-live-001"
    
    # 'Aoede' is often a good professional female voice, or 'Charon' for deep male.
    voice_id = os.getenv("GEMINI_LIVE_VOICE") or "Aoede" 

    gemini_params = GeminiLiveInputParams(temperature=0.3)
    # Force Arabic support setting if available in your SDK version, otherwise defaults work well
    try:
        gemini_params.language = Language.AR
    except:
        pass

    gemini_kwargs = {
        "api_key": api_key,
        "voice_id": voice_id,
        "system_instruction": system_prompt,
        "params": gemini_params,
        "inference_on_context_initialization": True,
        "model": model
    }
    
    gemini_live = GeminiLiveService(**gemini_kwargs)

    call_end_delay_s = 2.0
    lead_finalized = {"value": None}

    # ------------------------------------------------------------------
    # 4. Function Definitions (Tools)
    # ------------------------------------------------------------------
    async def confirm_appointment(params: FunctionCallParams):
        logger.info(f"TOOL: Confirming appointment for {lead_data['id']}")
        lead_finalized["value"] = "CONFIRMED"
        update_lead_status(lead_data["id"], "CONFIRMED")
        # We tell the LLM it succeeded so it can say "Great, see you then"
        await params.result_callback({"status": "success", "msg": "Appointment confirmed in system."})
        # Optional: Schedule hangup
        if call_control_id:
             asyncio.create_task(hangup_telnyx_call(call_control_id, 4.0))

    async def cancel_appointment(params: FunctionCallParams):
        logger.info(f"TOOL: Cancelling appointment for {lead_data['id']}")
        lead_finalized["value"] = "CANCELLED"
        reason = params.arguments.get("reason", "Patient request")
        update_lead_status(lead_data["id"], "CANCELLED")
        await params.result_callback({"status": "cancelled", "reason": reason})
        if call_control_id:
             asyncio.create_task(hangup_telnyx_call(call_control_id, 4.0))

    gemini_live.register_function("confirm_appointment", confirm_appointment)
    gemini_live.register_function("cancel_appointment", cancel_appointment)

    # ------------------------------------------------------------------
    # 5. Pipeline Assembly
    # ------------------------------------------------------------------
    mm_perf = MultimodalPerf()
    # Start with a hidden message to prime the bot to speak the greeting OR just wait for the user
    # Ideally, we want the bot to say the greeting first. 
    # Gemini Live often speaks first if we send a "User joined" or empty message, 
    # but here we rely on the system prompt instruction "Your first spoken line must be..."
    mm_context = LLMContext(messages=[{"role": "user", "content": "The call has connected. Say the greeting now."}])

    # Turn Strategies
    start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
    stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=0.6)] # Snappy turns
    
    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies),
        ),
    )

    # Init Context
    try:
        await gemini_live.set_context(mm_context)
    except Exception:
        pass

    # Processors
    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.7)
    transcript_fallback = AppointmentStatusTranscriptFallback(
        lead_id=lead_data["id"],
        call_control_id=call_control_id,
        call_end_delay_s=call_end_delay_s,
        finalized_ref=lead_finalized,
    )
    user_stop_trigger = MultimodalUserStopRunTrigger(delay_s=0.1, min_interval_s=0.5)
    
    pipeline = Pipeline(
        [
            transport.input(),
            InboundAudioLogger(),
            mm_aggregators.user(),
            user_stop_trigger,
            gemini_live,
            transcript_trigger,
            transcript_fallback,
            TurnStateLogger(),
            OutboundAudioLogger(),
            AudioFrameChunker(chunk_ms=20), # Smooth playback
            mm_perf,
            transport.output(),
            mm_aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    transcript_trigger.set_queue_frames(task.queue_frames)
    user_stop_trigger.set_queue_frames(task.queue_frames)
    
    runner = PipelineRunner()
    
    # ------------------------------------------------------------------
    # 6. Safety & Re-prompts
    # ------------------------------------------------------------------
    call_alive = {"value": True}

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client):
        call_alive["value"] = False
        transcript_trigger.cancel_pending()
        user_stop_trigger.cancel_pending()

    # If the bot is silent for too long at the start, nudge it
    async def kickstart_conversation():
        await asyncio.sleep(2.0)
        if not _MM.get("last_bot_started_ts") and call_alive["value"]:
            logger.info("Bot silent at start, triggering LLM...")
            await task.queue_frames([LLMRunFrame()])

    asyncio.create_task(kickstart_conversation())

    await runner.run(task)
