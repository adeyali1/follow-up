import os
import asyncio
import aiohttp
from urllib.parse import quote
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMRunFrame, LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
)
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    TranscriptionUserTurnStartStrategy,
    TranscriptionUserTurnStopStrategy,
)
from services.supabase_service import update_lead_status
import json
import time

_MM = {
    "last_user_transcription_ts": None,
    "last_bot_started_ts": None,
    "last_llm_run_ts": None,
}

BOT_BUILD_ID = "2026-01-20-ammani-natural"
_VAD_MODEL = {"value": None}

# ------------------------------------------------------------------
# 🔑 ARABIC NAME NORMALIZATION (FIXES PRONUNCIATION)
# ------------------------------------------------------------------
AR_NAME_MAP = {
    "oday": "عدي",
    "omar": "عمر",
    "ahmad": "أحمد",
    "mohammad": "محمد",
    "muhammad": "محمد",
    "ali": "علي",
    "sara": "سارة",
    "noor": "نور",
    "lina": "لينا",
    "yasmin": "ياسمين",
}

def normalize_customer_name_for_ar(name: str) -> str:
    if not name:
        return "عزيزي"
    key = name.strip().lower()
    return AR_NAME_MAP.get(key, name)

# ------------------------------------------------------------------
# TELNYX HANGUP
# ------------------------------------------------------------------
async def hangup_telnyx_call(call_control_id: str, delay_s: float):
    if not call_control_id:
        return
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    try:
        url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id)}/actions/hangup"
        headers = {
            "Authorization": f"Bearer {os.getenv('TELNYX_API_KEY')}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            await session.post(url, headers=headers, json={"reason": "normal_clearing"})
    except Exception:
        pass

# ------------------------------------------------------------------
# MAIN BOT
# ------------------------------------------------------------------
async def run_bot(websocket_client, lead_data, call_control_id=None):

    patient_name = normalize_customer_name_for_ar(
        lead_data.get("patient_name", "عزيزي")
    )

    treatment = lead_data.get("treatment", "تنظيف أسنان")
    appointment_time = lead_data.get("appointment_time", "الساعة 11")

    greeting_text = f"السلام عليكم، معك سارة من عيادة ابتسامة. معي {patient_name}؟"

    system_prompt = f"""
أنتِ سارة.
موظفة تنسيق مواعيد في عيادة أسنان بعمان.

أسلوبك:
- أردني عمّاني طبيعي 100%
- جُمَل قصيرة
- نبرة دافية
- بدون فصحى
- بدون تمثيل
- احكي زي ما بتحكي مع حدا عالتلفون

ممنوع:
- أسلوب إعلاني
- شعر
- مبالغة
- شرح طويل

ابدئي المكالمة بالجملة التالية فقط:
"{greeting_text}"

بعدها طبيعي اسألي:
"{patient_name}، بس بحب أأكد موعد {treatment} ع {appointment_time}، تمام؟"

إذا وافق:
- رد قصير
- نبرة مريحة
- سكّري المكالمة بلطف

إذا تردد:
- طمّني
- احكي بسطر واحد
"""

    # ------------------------------------------------------------------
    # GEMINI LIVE
    # ------------------------------------------------------------------
    from pipecat.services.google.gemini_live.llm import (
        GeminiLiveLLMService,
        InputParams,
    )

    gemini_params = InputParams(
        temperature=0.45,   # 👈 less robotic
        language=Language.AR,
        sample_rate=16000,
    )

    gemini_live = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-2.0-flash-live-001",
        voice_id="Charon",
        system_instruction=system_prompt,
        params=gemini_params,
        inference_on_context_initialization=True,
    )

    async def confirm_appointment(params: FunctionCallParams):
        update_lead_status(lead_data["id"], "CONFIRMED")
        await params.result_callback({"ok": True})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, 2))

    async def cancel_appointment(params: FunctionCallParams):
        update_lead_status(lead_data["id"], "CANCELLED")
        await params.result_callback({"ok": True})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, 2))

    gemini_live.register_function("update_lead_status_confirmed", confirm_appointment)
    gemini_live.register_function("update_lead_status_cancelled", cancel_appointment)

    serializer = TelnyxFrameSerializer(
        stream_id="stream",
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding="PCMU",
        inbound_encoding="PCMU",
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(min_volume=0.6, confidence=0.7)
            ),
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
    )

    context = LLMContext(messages=[{"role": "user", "content": "ابدئي"}])

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy()],
                stop=[TranscriptionUserTurnStopStrategy(timeout=0.8)],
            )
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            aggregators.user(),
            gemini_live,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()
    await runner.run(task)
