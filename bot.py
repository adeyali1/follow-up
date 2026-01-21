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
from services.supabase_service import update_lead_status
import json
import time

# --- Performance Tracking ---
_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-21-saudi-ahmed-pro-v2"
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
                logger.info(f"Latency: {now - user_ts:.3f}s")
        await self.push_frame(frame, direction)

class MultimodalTranscriptRunTrigger(FrameProcessor):
    """Triggers the LLM when a final transcript is ready if not already triggered by VAD."""
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
        if self._queue_frames is None or self._last_user_transcript_ts != ts:
            return
        # Avoid double trigger if VAD already handled it
        if _MM["last_llm_run_ts"] and (time.monotonic() - _MM["last_llm_run_ts"] < 0.4):
            return
        _MM["last_llm_run_ts"] = time.monotonic()
        await self._queue_frames([LLMRunFrame()])
    def cancel_pending(self):
        if self._pending:
            self._pending.cancel()
            self._pending = None
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            if not getattr(frame, "is_final", True):
                return
            now = time.monotonic()
            self._last_user_transcript_ts = now
            self.cancel_pending()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

class AudioFrameChunker(FrameProcessor):
    """Optimized chunking to prevent robotic stuttering."""
    def __init__(self, *, chunk_ms: int = 20): # Lowered to 20ms for smoother flow
        super().__init__()
        self._chunk_ms = chunk_ms
    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame) and self._chunk_ms > 0:
            audio = frame.audio
            sample_rate = frame.sample_rate
            bytes_per_sample = 2
            chunk_size = int(sample_rate * self._chunk_ms / 1000) * bytes_per_sample
            if len(audio) > chunk_size:
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i : i + chunk_size]
                    await self.push_frame(AudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=frame.num_channels), direction)
                return
        await self.push_frame(frame, direction)

class LeadStatusTranscriptFallback(FrameProcessor):
    def __init__(self, lead_id, call_control_id, finalized_ref):
        super().__init__()
        self._lead_id = lead_id
        self._call_control_id = call_control_id
        self._finalized = finalized_ref
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user" and getattr(frame, "is_final", True):
            text = (getattr(frame, "text", "") or "").lower()
            if any(x in text for x in ["تمام", "ماشي", "اوكي", "حجز"]):
                update_lead_status(self._lead_id, "CONFIRMED")
            elif any(x in text for x in ["الغ", "كنسل", "بطلت"]):
                update_lead_status(self._lead_id, "CANCELLED")
        await self.push_frame(frame, direction)

def normalize_gemini_live_model_name(model: str) -> str:
    return "models/gemini-2.0-flash-exp"

async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id: return
    await asyncio.sleep(delay_s)
    url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id, safe='')}/actions/hangup"
    headers = {"Authorization": f"Bearer {os.getenv('TELNYX_API_KEY')}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, headers=headers, json={"reason": "normal_clearing"})
    except Exception: pass

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting Pro Saudi Bot: {lead_data.get('id')}")
    
    # 1. Telnyx Handshake
    stream_id = "telnyx_stream"
    inbound_encoding = "PCMU"
    try:
        msg_text = await websocket_client.receive_text()
        msg = json.loads(msg_text)
        stream_id = msg.get("stream_id") or msg.get("data", {}).get("stream_id") or "telnyx_stream"
    except Exception: pass

    # 2. VAD - Optimized for Barge-in (Listening)
    # Lower start_secs (0.1) means bot stops talking FASTER when user speaks
    vad = SileroVADAnalyzer(params=VADParams(
        min_volume=0.5, 
        start_secs=0.1, 
        stop_secs=0.3, 
        confidence=0.65
    ))

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding=inbound_encoding,
        inbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(sample_rate=16000),
    )
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
    )

    # 3. System Prompt - Enhanced for Interruption Handling
    system_prompt = """أنت "أحمد"، خبير مبيعات محترف من "موصول" (Mawsool at mawsool.tech). لهجتك سعودية (نجدية/بيضاء) قوية وطبيعية جداً.

**قواعد ذهبية للتعامل:**
1. **اسمع أكثر مما تتكلم:** إذا قاطعك العميل أو بدأ يتكلم، اصمت فوراً (Barge-in). لا تكمل جملتك إذا قاطعك.
2. **لا تكن روبوت:** استخدم كلمات مثل "يا طويل العمر"، "سم"، "أبشر"، "هلا بك".
3. **التفاعل الطبيعي:** إذا سألك "من معي؟" جاوب فوراً ولا تكمل السكريبت القديم. 
4. **الاختصار:** جملك لا تزيد عن 10 كلمات.

**الهدف:** حجز ديمو (Demo) لمنصة موصول (قاعدة بيانات B2B).
رابط الديمو: https://calendar.app.google/EutcxTBmd9zRtYrv9

**سير المكالمة:**
- الترحيب: "هلا والله! معك أحمد من موصول، عساك طيب؟"
- إذا قال مشغول: "ما أطول عليك، بس دقيقة أعلمك كيف نوفر عليك آلاف الريالات في البحث عن العملاء."
- البيع: "نعطيك أرقام وإيميلات مدراء الشركات مباشرة بدقة 98٪."
- القفل: "متى يناسبك نسوي ديمو سريع وتجرب بنفسك؟" """

    # 4. Gemini Live Configuration
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
    
    gemini_live = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        voice_id="Aoede", # Best female voice, or use "Charon" for male
        model="models/gemini-2.0-flash-exp",
        system_instruction=system_prompt,
        params=InputParams(temperature=0.7, sample_rate=16000),
        inference_on_context_initialization=True
    )

    # 5. Tools
    lead_finalized = {"value": None}
    async def update_status(params: FunctionCallParams):
        status = "CONFIRMED" if "confirmed" in params.name else "CANCELLED"
        lead_finalized["value"] = status
        update_lead_status(lead_data.get("id", "mock"), status)
        await params.result_callback({"status": "updated"})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, 2.5))

    gemini_live.register_function("update_lead_status_confirmed", update_status)
    gemini_live.register_function("update_lead_status_cancelled", update_status)

    # 6. Turn Management - FIX FOR "NOT LISTENING"
    # We use VADUserTurnStartStrategy to ensure the bot stops the MOMENT it hears the user.
    try:
        from pipecat.turns.user_start import VADUserTurnStartStrategy
        start_strategy = VADUserTurnStartStrategy(enable_interruptions=True)
    except Exception:
        start_strategy = TranscriptionUserTurnStartStrategy(use_interim=True)

    mm_aggregators = LLMContextAggregatorPair(
        LLMContext(messages=[{"role": "user", "content": "ابدأ المكالمة بلهجة سعودية حيوية"}]),
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[start_strategy],
                stop=[TranscriptionUserTurnStopStrategy(timeout=0.6)]
            )
        ),
    )

    # 7. Pipeline Construction
    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.5)
    
    pipeline = Pipeline([
        transport.input(),              # Audio in from Telnyx
        mm_aggregators.user(),          # Handle user turns / Interruption detection
        gemini_live,                    # Gemini Multimodal Live
        transcript_trigger,             # Fallback trigger
        LeadStatusTranscriptFallback(lead_data.get('id'), call_control_id, lead_finalized),
        AudioFrameChunker(chunk_ms=20), # Smooth audio out
        mm_perf,                        # Monitor latency
        transport.output(),             # Audio out to Telnyx
        mm_aggregators.assistant(),     # Aggregate bot responses
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    transcript_trigger.set_queue_frames(task.queue_frames)

    runner = PipelineRunner()
    
    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client):
        transcript_trigger.cancel_pending()

    # Failsafe initial trigger
    async def initial_trigger():
        await asyncio.sleep(1.5)
        if _MM["last_bot_started_ts"] is None:
            await task.queue_frames([LLMRunFrame()])

    asyncio.create_task(initial_trigger())
    await runner.run(task)
