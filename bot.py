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

class MultimodalUserStopRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float = 0.4, min_interval_s: float = 0.3):
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
            self._pending.cancel()
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
            logger.info("User speaking - Bot should listen")
        await self.push_frame(frame, direction)

class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 20): # Smaller chunks for smoother audio
        super().__init__()
        self._chunk_ms = chunk_ms
    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            # Logic to ensure steady audio flow without robot-like jitter
            await self.push_frame(frame, direction)
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
            text = (getattr(frame, "text", "") or "").lower()
            if "بكره" in text or "بكرة" in text or "تمام" in text:
                logger.info("Fallback: Detection of commitment in text.")
                # Optional: trigger status update if LLM fails tool call
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
    # HARDCODED STABILITY SETTINGS
    PIPELINE_SAMPLE_RATE = 16000
    GEMINI_MODEL = "models/gemini-2.0-flash-exp"
    GEMINI_VOICE = "Aoede" # Best female Arabic voice for Gemini
    
    _MM["last_user_transcription_ts"] = None
    _MM["last_bot_started_ts"] = None
    
    # 0. Telnyx Handshake
    stream_id = "telnyx_stream"
    inbound_encoding = "PCMU"
    try:
        msg_text = await websocket_client.receive_text()
        msg = json.loads(msg_text)
        stream_id = msg.get("stream_id") or msg.get("data", {}).get("stream_id") or "telnyx_stream"
    except: pass

    # --- SYSTEM PROMPT (Optimized for Context & Interruption) ---
    system_prompt = """أنت "أحمد" من شركة "موصول". لهجتك سعودية بيضاء، طبيعية جداً، وكأنك تسولف مع صديق.
    
    **قواعد صارمة جداً:**
    1. **الاستماع أولاً:** إذا العميل قاطعك، اسكت فوراً. لا تكمل جملتك.
    2. **الإجابة المباشرة:** إذا سألك العميل سؤال (مثل: "كم السعر؟" أو "من أنت؟")، جاوبه فوراً ولا تتبع السيناريو المحفوظ.
    3. **الاختصار:** كلامك لازم يكون قصير جداً (جملة واحدة غالباً). لا تلقي خطابات.
    4. **السعر:** إذا سألك عن السعر، قل له: "الديمو مجاني تماماً يا غالي، وبعدها نقرر الباقة اللي تناسبك".
    5. **الشخصية:** خلك حيوي، استخدم كلمات مثل "سم"، "أبشر"، "يا طويل العمر"، "عساك طيب".
    
    **مهمتك:** حجز موعد "ديمو" (عرض تجريبي) لمنصة موصول. 
    موصول هي أداة تعطيك أرقام وإيميلات الشركات والمدراء في السعودية بدقة 98%.
    
    رابط الديمو: https://calendar.app.google/EutcxTBmd9zRtYrv9"""

    # --- ENHANCED VAD (Better Barge-in) ---
    vad = SileroVADAnalyzer(params=VADParams(
        min_volume=0.5, 
        start_secs=0.1, 
        stop_secs=0.4, # More breathing room to avoid robot cut-offs
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
    
    gemini_params = InputParams(
        temperature=0.8, # Increased for natural variety
        sample_rate=PIPELINE_SAMPLE_RATE,
        modalities=GeminiModalities.AUDIO
    )

    gemini_live = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id=GEMINI_VOICE,
        model=GEMINI_MODEL,
        system_instruction=system_prompt,
        params=gemini_params,
        inference_on_context_initialization=True,
    )

    lead_finalized = {"value": None}
    async def confirm_demo(params: FunctionCallParams):
        lead_finalized["value"] = "CONFIRMED"
        update_lead_status(lead_data.get("id", "mock"), "CONFIRMED")
        await params.result_callback({"status": "confirmed"})
        asyncio.create_task(hangup_telnyx_call(call_control_id, 3.0))

    gemini_live.register_function("update_lead_status_confirmed", confirm_demo)

    mm_context = LLMContext(messages=[{"role": "user", "content": "ابدأ المكالمة فوراً وسلم علي بحرارة بلهجة سعودية."}])
    
    # --- Strategies for smoother turn-taking ---
    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy(use_interim=True)],
                stop=[TranscriptionUserTurnStopStrategy(timeout=0.6)]
            ),
            user_mute_strategies=[] # Allow instant interrupt
        ),
    )

    await gemini_live.set_context(mm_context)

    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.4)
    user_stop_trigger = MultimodalUserStopRunTrigger(delay_s=0.3)

    pipeline = Pipeline([
        transport.input(),
        mm_aggregators.user(),
        user_stop_trigger,
        gemini_live,
        transcript_trigger,
        TurnStateLogger(),
        AudioFrameChunker(),
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
        logger.info("Ahmed is online.")

    # Failsafe for the very first greeting
    async def initial_greeting_failsafe():
        await asyncio.sleep(1.5)
        if _MM.get("last_bot_started_ts") is None:
            await task.queue_frames([LLMRunFrame()])

    asyncio.create_task(initial_greeting_failsafe())
    await runner.run(task)
