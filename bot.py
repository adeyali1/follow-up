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

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-21-saudi-ahmed-v3-fixed"
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
        _MM["last_llm_run_ts"] = time.monotonic()
        await self._queue_frames([LLMRunFrame()])
    def cancel_pending(self):
        if self._pending:
            try: self._pending.cancel()
            except: pass
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
        if self._pending:
            try: self._pending.cancel()
            except: pass
            self._pending = None
    async def _schedule(self, ts: float):
        try:
            await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError:
            return
        if self._queue_frames is None or self._last_stop_ts != ts:
            return
        last_run = _MM.get("last_llm_run_ts")
        now = time.monotonic()
        if last_run and (now - last_run) < self._min_interval_s:
            return
        _MM["last_llm_run_ts"] = now
        await self._queue_frames([LLMRunFrame()])
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if type(frame).__name__ == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            self._last_stop_ts = now
            self.cancel_pending()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

class TurnStateLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "UserStartedSpeakingFrame":
            logger.info("Turn: user started speaking")
        elif frame_name in {"BotStartedSpeakingFrame", "TTSStartedFrame"}:
            logger.info("Turn: bot speaking")
        await self.push_frame(frame, direction)

class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 40):
        super().__init__()
        self._chunk_ms = int(chunk_ms)
        self._pace = True
    async def process_frame(self, frame, direction):
        # CRITICAL FIX: Always call super().process_frame to avoid StartFrame errors
        await super().process_frame(frame, direction)
        
        if self._chunk_ms > 0 and isinstance(frame, AudioRawFrame):
            audio = frame.audio
            if isinstance(audio, (bytes, bytearray)) and len(audio) > 0:
                sample_rate = int(getattr(frame, "sample_rate", 16000))
                channels = int(getattr(frame, "num_channels", 1))
                bytes_per_sample = 2
                bytes_per_frame = channels * bytes_per_sample
                chunk_bytes = int(sample_rate * self._chunk_ms / 1000) * bytes_per_frame
                
                if len(audio) > chunk_bytes:
                    for i in range(0, len(audio), chunk_bytes):
                        chunk = audio[i : i + chunk_bytes]
                        if chunk:
                            await self.push_frame(AudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=channels), direction)
                    return
        await self.push_frame(frame, direction)

class LeadStatusTranscriptFallback(FrameProcessor):
    def __init__(self, lead_id, call_control_id, call_end_delay_s, finalized_ref):
        super().__init__()
        self._lead_id = lead_id
        self._call_control_id = call_control_id
        self._call_end_delay_s = call_end_delay_s
        self._finalized_ref = finalized_ref

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            if not getattr(frame, "is_final", True): return
            text = (getattr(frame, "text", "") or "").lower()
            if any(x in text for x in ["تمام", "ماشي", "اوكي", "بكره", "بكرة"]):
                logger.info("Fallback: Detection of commitment in text")
        await self.push_frame(frame, direction)

async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id: return
    telnyx_key = os.getenv("TELNYX_API_KEY")
    await asyncio.sleep(delay_s)
    url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id, safe='')}/actions/hangup"
    headers = {"Authorization": f"Bearer {telnyx_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, headers=headers, json={"reason": "normal_clearing"})
    except: pass

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting Saudi Sales Bot for lead: {lead_data.get('id')}")
    
    # HARDCODED PROMPT & CONFIG
    PIPELINE_SAMPLE_RATE = 16000
    
    system_prompt = """أنت "أحمد"، خبير مبيعات محترف من "موصول". لهجتك سعودية (نجدية/بيضاء) قوية وطبيعية جداً.

**أسلوبك:**
- صوتك مليء بالطاقة والثقة، بس بدون تكلف.
- ردودك **قصيرة جداً وموجزة** (جملة أو جملتين). لا تطول في الكلام أبداً.
- تكلم زي الناس الطبيعيين: استخدم كلمات عامية مثل "هلا ابوي"، "سم"، "يا طويل العمر"، "أبشر"، "على خشمي".
- **إذا قاطعك العميل، اسكت فوراً واسمعه.**

**الهدف:**
تحجز موعد لديمو (Demo) لمنصة موصول. موصول هي قاعدة بيانات ضخمة للشركات (B2B Leads).

**طريقة الكلام:**
1. أول ما يرد: "هلا والله! معك أحمد من شركة موصول، عساك طيب؟" (خلك سريع وعفوي).
2. إذا قال مشغول: "ما أطول عليك طال عمرك، بس دقيقة وحدة وتعرف كيف نوفر عليك آلاف الريالات. تسمح لي؟"
3. إذا سألك عن السعر: "الديمو مجاني طال عمرك، بس عشان تشوف المنصة وتجربها بنفسك، متى يناسبك؟"

**ملاحظات مهمة:**
- لا تتكلم فصحى أبداً.
- الهدف هو "الديمو"، مو البيع في التلفون."""

    # 0. Telnyx Handshake
    stream_id = "telnyx_stream"
    inbound_encoding = "PCMU"
    try:
        msg_text = await websocket_client.receive_text()
        msg = json.loads(msg_text)
        stream_id = msg.get("stream_id") or msg.get("data", {}).get("stream_id") or "telnyx_stream"
    except: pass

    # ENHANCED VAD
    vad = SileroVADAnalyzer(params=VADParams(
        min_volume=0.5, 
        start_secs=0.1, 
        stop_secs=0.4, # Improved listening
        confidence=0.6
    ))

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding=inbound_encoding,
        inbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(sample_rate=PIPELINE_SAMPLE_RATE),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
        ),
    )

    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams, GeminiModalities
    
    gemini_live = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Aoede",
        model="models/gemini-2.0-flash-exp",
        system_instruction=system_prompt,
        params=InputParams(temperature=0.7, modalities=GeminiModalities.AUDIO),
        inference_on_context_initialization=True,
    )

    # Tool registration
    lead_finalized = {"value": None}
    async def confirm_appointment(params: FunctionCallParams):
        lead_finalized["value"] = "CONFIRMED"
        update_lead_status(lead_data.get("id"), "CONFIRMED")
        await params.result_callback({"value": "Confirmed."})
        if call_control_id: asyncio.create_task(hangup_telnyx_call(call_control_id, 2.5))

    gemini_live.register_function("update_lead_status_confirmed", confirm_appointment)

    # Context to trigger the Saudi Greeting
    mm_context = LLMContext(messages=[{"role": "user", "content": "ابدأ المكالمة فوراً وسلم علي بلهجة سعودية."}])
    
    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy(use_interim=True)],
                stop=[TranscriptionUserTurnStopStrategy(timeout=0.6)]
            )
        ),
    )

    await gemini_live.set_context(mm_context)

    # Trigger components
    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.7)
    user_stop_trigger = MultimodalUserStopRunTrigger()
    transcript_fallback = LeadStatusTranscriptFallback(lead_data.get("id"), call_control_id, 2.5, lead_finalized)

    # PIPELINE IN ORIGINAL ORDER
    pipeline = Pipeline([
        transport.input(),
        mm_aggregators.user(),
        user_stop_trigger,
        gemini_live,
        transcript_trigger,
        transcript_fallback,
        TurnStateLogger(),
        AudioFrameChunker(chunk_ms=40),
        MultimodalPerf(),
        transport.output(),
        mm_aggregators.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    transcript_trigger.set_queue_frames(task.queue_frames)
    user_stop_trigger.set_queue_frames(task.queue_frames)

    runner = PipelineRunner()
    
    @transport.event_handler("on_client_connected")
    async def _on_client_connected(_transport, _client):
        logger.info("Ahmed (Saudi Bot) is connected and ready.")

    # Greeting Failsafe
    async def initial_failsafe():
        await asyncio.sleep(2.0)
        if _MM.get("last_bot_started_ts") is None:
            logger.warning("Failsafe greeting triggered")
            await task.queue_frames([LLMRunFrame()])

    asyncio.create_task(initial_failsafe())
    await runner.run(task)
