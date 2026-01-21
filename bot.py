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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy
import json
import time

# --- MOCK DB ---
def update_lead_status(lead_id, status):
    logger.info(f"--- DATABASE UPDATE: LEAD {lead_id} STATUS -> {status} ---")

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None}
_VAD_MODEL = {"value": None}

# --- 1. LATENCY MONITOR ---
class MultimodalPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if type(frame).__name__ == "TranscriptionFrame" and getattr(frame, "role", None) == "user":
             _MM["last_user_transcription_ts"] = time.monotonic()
        if type(frame).__name__ in {"BotStartedSpeakingFrame", "TTSStartedFrame"}:
            now = time.monotonic()
            _MM["last_bot_started_ts"] = now
            if _MM["last_user_transcription_ts"]:
                latency = (now - _MM['last_user_transcription_ts']) * 1000
                logger.info(f"⚡ Latency: {latency:.0f}ms")
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

# --- 2. OPTIMIZED VOIP CHUNKER (20ms) ---
# This is critical for Telnyx. Do not change 20ms.
class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 20):
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
            
            # Calculate exactly how many bytes equal 20ms
            target_bytes = int(sample_rate * self._chunk_ms / 1000) * bytes_per_frame
            target_bytes = max(target_bytes - (target_bytes % bytes_per_frame), bytes_per_frame)
            
            if len(frame.audio) > target_bytes:
                audio = frame.audio
                frame_type = type(frame)
                first = True
                for i in range(0, len(audio), target_bytes):
                    chunk = audio[i : i + target_bytes]
                    if not chunk: continue
                    await self.push_frame(frame_type(audio=chunk, sample_rate=sample_rate, num_channels=channels), direction)
                    if self._pace and not first:
                        # Exact sleep to mimic real-time streaming
                        await asyncio.sleep(self._chunk_ms / 1000.0)
                    first = False
                return
        await self.push_frame(frame, direction)

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

def normalize_gemini_live_model_name(model: str) -> str:
    if not model or not model.strip(): return "models/gemini-2.0-flash-exp"
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

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting OPTIMIZED bot for lead: {lead_data['id']}")
    
    # AI works best at 16k. We will downsample for Telnyx later.
    pipeline_sample_rate = 16000 
    
    # 0. Telnyx Handshake & Codec Detection
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU" # Default Fallback
    
    try:
        for _ in range(3):
            msg_text = await websocket_client.receive_text()
            logger.debug(f"Telnyx Handshake: {msg_text}")
            msg = json.loads(msg_text)
            
            # Smart Codec Detection
            media_format = msg.get("media_format") or msg.get("start", {}).get("media_format")
            if media_format:
                enc = media_format.get("encoding", "").upper()
                if enc == "PCMA": 
                    inbound_encoding = "PCMA"
                    logger.info("Codec detected: PCMA (Optimized for Jordan/EU)")
                elif enc == "PCMU":
                    inbound_encoding = "PCMU"

            if "stream_id" in msg: 
                stream_id = msg["stream_id"]
                break
            if msg.get("event") == "start" and "stream_id" in msg:
                stream_id = msg["stream_id"]
                break
    except Exception as e:
        logger.error(f"Handshake error: {e}")

    # 1. Config
    customer_name = lead_data.get("customer_name") or "عزيزي"
    appointment_type = lead_data.get("appointment_type", "تنظيف وتبييض أسنان")
    appointment_time = lead_data.get("appointment_time", "بكرا الساعة 4 العصر")

    # --- 3. PROFESSIONAL VAD TUNING ---
    # Optimized to ignore background noise but catch human speech quickly
    vad = SileroVADAnalyzer(
        params=VADParams(
            min_volume=0.8,    # High threshold ignores breath/noise
            start_secs=0.4,    # Waits 400ms to confirm speech (anti-interruption)
            stop_secs=0.7,     # Snappy turn taking
            confidence=0.75
        )
    )

    # Serializer matches Telnyx codec exactly to prevent transcoding lag
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

    # 4. System Prompt (Token Efficient)
    system_prompt = f"""
# ROLE
You are **Sara** from **"Elite Dental Clinic"** in Amman.
Confirm appointment with {customer_name}.

# TONE (Ammani Arabic)
- Professional, Warm.
- NO slang, NO Fusha. Use "حضرتك", "يا هلا".

# CONTEXT
- Treatment: {appointment_type}
- Time: {appointment_time}
- Loc: Abdoun.

# WORKFLOW
1. **Greeting:** "مرحبا {customer_name}، معك سارة من عيادة إليت. سامعني؟"
2. **Confirm:** "بأكد موعدك {appointment_time} لـ {appointment_type}."
3. **If Confirmed:** Tool `confirm_appointment`.
4. **If Cancel:** Tool `cancel_appointment`.

# CRITICAL RULES
- START SPEAKING IMMEDIATELY.
- Keep replies under 5 seconds.
"""

    # 5. Gemini Service (Fast Start)
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
    
    gemini_live = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        voice_id=os.getenv("GEMINI_LIVE_VOICE") or "Aoede",
        system_instruction=system_prompt,
        params=InputParams(temperature=0.3),
        # TRUE = Generates audio immediately (Zero Latency Start)
        inference_on_context_initialization=True, 
        model=normalize_gemini_live_model_name(os.getenv("GEMINI_LIVE_MODEL"))
    )

    lead_finalized = {"value": None}

    # 6. Tools
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

    # 7. Pipeline Assembly
    mm_perf = MultimodalPerf()
    # Empty context relies on inference_on_context_initialization
    mm_context = LLMContext(messages=[]) 

    start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
    stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=0.7)]
    
    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies),
        ),
    )

    await gemini_live.set_context(mm_context)

    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.6)
    transcript_fallback = AppointmentStatusTranscriptFallback(
        lead_id=lead_data["id"],
        call_control_id=call_control_id,
        finalized_ref=lead_finalized,
    )
    
    # 20ms Chunker (The most important line for audio quality)
    audio_chunker = AudioFrameChunker(chunk_ms=20)

    pipeline = Pipeline(
        [
            transport.input(),
            mm_aggregators.user(),
            gemini_live,
            transcript_trigger,
            transcript_fallback,
            audio_chunker, # Must be before transport output
            mm_perf,
            transport.output(),
            mm_aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    transcript_trigger.set_queue_frames(task.queue_frames)
    
    runner = PipelineRunner()
    await runner.run(task)
