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
from pipecat.frames.frames import LLMRunFrame, LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy

import json
import time

# --- MOCK DB ---
def update_lead_status(lead_id, status):
    logger.info(f"--- DATABASE UPDATE: LEAD {lead_id} STATUS -> {status} ---")

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
_VAD_MODEL = {"value": None}

# --- PERFORMANCE LOGGING ---
class MultimodalPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "TranscriptionFrame" and getattr(frame, "role", None) == "user":
             _MM["last_user_transcription_ts"] = time.monotonic()
        if frame_name in {"BotStartedSpeakingFrame", "TTSStartedFrame"}:
            now = time.monotonic()
            _MM["last_bot_started_ts"] = now
        await self.push_frame(frame, direction)

# --- RE-TRIGGER LOGIC (KEEPS CONVERSATION ALIVE) ---
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
            if self._queue_frames and self._last_user_transcript_ts == ts:
                await self._queue_frames([LLMRunFrame()])
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            if getattr(frame, "is_final", False):
                now = time.monotonic()
                self._last_user_transcript_ts = now
                if self._pending: self._pending.cancel()
                self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

class MultimodalUserStopRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float = 0.05):
        super().__init__()
        self._delay_s = delay_s
        self._queue_frames = None
        self._pending = None
        self._last_stop_ts = None

    def set_queue_frames(self, queue_frames):
        self._queue_frames = queue_frames

    async def _schedule(self, ts: float):
        try:
            await asyncio.sleep(self._delay_s)
            if self._queue_frames and self._last_stop_ts == ts:
                await self._queue_frames([LLMRunFrame()])
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if type(frame).__name__ == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            self._last_stop_ts = now
            if self._pending: self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

# --- AUDIO SMOOTHING (CRITICAL FOR QUALITY) ---
class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 60): # Increased to 60ms for smoothness
        super().__init__()
        self._chunk_ms = chunk_ms
        self._pace = True

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and len(frame.audio) > 0:
            sample_rate = frame.sample_rate
            channels = frame.num_channels
            bytes_per_sample = 2
            bytes_per_frame = channels * bytes_per_sample
            chunk_bytes = int(sample_rate * self._chunk_ms / 1000) * bytes_per_frame
            # Align to frame boundary
            chunk_bytes = max(chunk_bytes - (chunk_bytes % bytes_per_frame), bytes_per_frame)
            
            if len(frame.audio) > chunk_bytes:
                audio = frame.audio
                frame_type = type(frame)
                first = True
                for i in range(0, len(audio), chunk_bytes):
                    chunk = audio[i : i + chunk_bytes]
                    if not chunk: continue
                    await self.push_frame(frame_type(audio=chunk, sample_rate=sample_rate, num_channels=channels), direction)
                    if self._pace and not first:
                        await asyncio.sleep(self._chunk_ms / 1000.0)
                    first = False
                return
        await self.push_frame(frame, direction)

# --- FALLBACK TRANSCRIPT ANALYSIS ---
class AppointmentStatusTranscriptFallback(FrameProcessor):
    def __init__(self, *, lead_id: str, call_control_id: str | None, finalized_ref: dict):
        super().__init__()
        self._lead_id = lead_id
        self._call_control_id = call_control_id
        self._finalized_ref = finalized_ref

    @staticmethod
    def _is_confirm(text: str) -> bool:
        t = text.lower()
        if any(x in t for x in ["cancel", "no", "لا", "ما بدي", "الغ"]): return False
        return any(x in t for x in ["تمام", "ماشي", "أكيد", "موافق", "ان شاء الله", "جاي", "ok", "yes", "confirm"])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user" and getattr(frame, "is_final", False):
            if not self._finalized_ref.get("value"):
                text = getattr(frame, "text", "") or ""
                if self._is_confirm(text):
                    logger.info("Fallback: Transcript indicates confirmation.")
        await self.push_frame(frame, direction)

# --- UTILS ---
def normalize_gemini_live_model_name(model: str) -> str:
    if not model or not model.strip(): return "models/gemini-2.0-flash-live-001"
    if "models/" not in model: return f"models/{model}"
    return model

async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id or not os.getenv("TELNYX_API_KEY"): return
    if delay_s > 0: await asyncio.sleep(delay_s)
    url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id, safe='')}/actions/hangup"
    headers = {"Authorization": f"Bearer {os.getenv('TELNYX_API_KEY')}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"reason": "normal_clearing"}): pass
    except: pass

# --- MAIN BOT LOGIC ---
async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting CRYSTAL CLEAR bot for lead: {lead_data['id']}")
    
    pipeline_sample_rate = 16000 # Keep 16k for AI quality
    stream_sample_rate = 8000 # Telnyx standard
    
    # 0. Telnyx Handshake
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU"
    
    try:
        for _ in range(3):
            msg_text = await websocket_client.receive_text()
            msg = json.loads(msg_text)
            if "media_format" in msg:
                if msg["media_format"].get("encoding", "").upper() == "PCMA": inbound_encoding = "PCMA"
            elif "start" in msg:
                if msg["start"].get("media_format", {}).get("encoding", "").upper() == "PCMA": inbound_encoding = "PCMA"
            
            if "stream_id" in msg: 
                stream_id = msg["stream_id"]
                break
            if msg.get("event") == "start":
                if "stream_id" in msg: stream_id = msg["stream_id"]
                break
    except Exception:
        logger.warning("Handshake issue, proceeding with defaults")

    # 1. Config
    customer_name = lead_data.get("customer_name") or "عزيزي"
    appointment_type = lead_data.get("appointment_type", "تنظيف وتبييض أسنان")
    appointment_time = lead_data.get("appointment_time", "بكرا الساعة 4 العصر")

    # --- TUNED VAD FOR NOISE IMMUNITY ---
    # increased min_volume to 0.8 (filters background noise)
    # increased start_secs to 0.3 (filters quick random sounds)
    vad = SileroVADAnalyzer(
        params=VADParams(
            min_volume=0.8, 
            start_secs=0.3, 
            stop_secs=0.8, 
            confidence=0.75
        )
    )

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

    # 2. System Prompt
    system_prompt = f"""
# ROLE
You are **Sara**, Treatment Coordinator at **"Elite Dental Clinic"** in Amman.
Confirm appointment with {customer_name}.

# TONE (Ammani Arabic)
- Professional, Warm, Clear.
- NO slurring, NO slang, NO Fusha.
- Use "حضرتك", "يا هلا", "تمام".

# CONTEXT
- Treatment: {appointment_type}
- Time: {appointment_time}
- Loc: Abdoun.

# WORKFLOW
1. **Greeting:** "مرحبا {customer_name}، معك سارة من عيادة إليت للأسنان. سامعني واضح؟" (Wait for reply).
2. **Confirm:** "بأكد موعدك {appointment_time} لـ {appointment_type}. بانتظارك دكتور أسامة."
3. **If Confirmed:** Call tool `confirm_appointment` -> "ممتاز، نتشرف فيك."
4. **If Cancel:** Call tool `cancel_appointment` -> "ولا يهمك، بنرتب وقت تاني."

# RULES
- Speak immediately.
- Short sentences.
"""

    # 3. Gemini Service
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
    
    gemini_live = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        voice_id=os.getenv("GEMINI_LIVE_VOICE") or "Aoede",
        system_instruction=system_prompt,
        params=InputParams(temperature=0.3),
        inference_on_context_initialization=True,
        model=normalize_gemini_live_model_name(os.getenv("GEMINI_LIVE_MODEL"))
    )

    lead_finalized = {"value": None}

    # 4. Tools
    async def confirm_appointment(params: FunctionCallParams):
        lead_finalized["value"] = "CONFIRMED"
        update_lead_status(lead_data["id"], "CONFIRMED")
        await params.result_callback({"status": "success"})
        if call_control_id: asyncio.create_task(hangup_telnyx_call(call_control_id, 3.0))

    async def cancel_appointment(params: FunctionCallParams):
        lead_finalized["value"] = "CANCELLED"
        update_lead_status(lead_data["id"], "CANCELLED")
        await params.result_callback({"status": "cancelled"})
        if call_control_id: asyncio.create_task(hangup_telnyx_call(call_control_id, 3.0))

    gemini_live.register_function("confirm_appointment", confirm_appointment)
    gemini_live.register_function("cancel_appointment", cancel_appointment)

    # 5. Pipeline
    mm_perf = MultimodalPerf()
    # Hidden context message to seed the conversation
    mm_context = LLMContext(messages=[{"role": "user", "content": "Call connected. Greet the patient immediately."}])

    # --- TURN STRATEGIES (STRICTER) ---
    # Using Transcription Strategy but relying on the stricter VAD settings passed to Transport
    start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
    stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=0.8)] # Increased timeout to prevent cutting user off
    
    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies),
        ),
    )

    await gemini_live.set_context(mm_context)

    # Triggers
    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.7)
    transcript_fallback = AppointmentStatusTranscriptFallback(
        lead_id=lead_data["id"],
        call_control_id=call_control_id,
        finalized_ref=lead_finalized,
    )
    # Less aggressive user stop trigger
    user_stop_trigger = MultimodalUserStopRunTrigger(delay_s=0.2)
    
    # --- SMOOTHING ---
    # Increased chunk_ms to 60 for better audio stability over network
    audio_chunker = AudioFrameChunker(chunk_ms=60)

    pipeline = Pipeline(
        [
            transport.input(),
            mm_aggregators.user(),
            user_stop_trigger,
            gemini_live,
            transcript_trigger,
            transcript_fallback,
            audio_chunker, # Chunking before output
            mm_perf,
            transport.output(),
            mm_aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    transcript_trigger.set_queue_frames(task.queue_frames)
    user_stop_trigger.set_queue_frames(task.queue_frames)
    
    runner = PipelineRunner()
    
    # --- FAST START LOGIC ---
    @transport.event_handler("on_client_connected")
    async def _on_client_connected(_transport, _client):
        # As soon as audio connects, force the LLM to run. 
        # This is faster than waiting for a silence timeout.
        logger.info("Client connected. Triggering greeting immediately.")
        await task.queue_frames([LLMRunFrame()])

    await runner.run(task)
