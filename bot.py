# ===========================
# DENTAL TREATMENT COORDINATOR
# Native Jordanian Arabic
# Gemini Live Native Audio
# ===========================

import os
import asyncio
import aiohttp
import json
import time
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
from pipecat.frames.frames import (
    LLMRunFrame,
    LLMMessagesAppendFrame,
    AudioRawFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    TranscriptionUserTurnStartStrategy,
    TranscriptionUserTurnStopStrategy,
)
from pipecat.processors.frame_processor import FrameProcessor

# ===========================
# GLOBALS
# ===========================

BOT_BUILD_ID = "2026-01-21-native-jordanian-dental"
_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None}
_VAD_MODEL = {"value": None}


# ===========================
# UTILITIES
# ===========================

def normalize_arabic_name(name: str) -> str:
    if not name:
        return "عزيزي"
    n = name.strip().lower()
    if n in ["oday", "uday", "odai"]:
        return "عُدي"
    return name.strip()


async def hangup_telnyx_call(call_control_id: str, delay_s: float):
    if not call_control_id:
        return
    await asyncio.sleep(delay_s)
    url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id)}/actions/hangup"
    headers = {
        "Authorization": f"Bearer {os.getenv('TELNYX_API_KEY')}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        await session.post(url, headers=headers, json={"reason": "normal_clearing"})


# ===========================
# BOT
# ===========================

async def run_bot(websocket_client, lead_data, call_control_id=None):

    logger.info(f"Starting Dental Coordinator bot {BOT_BUILD_ID}")

    patient_name = normalize_arabic_name(
        lead_data.get("patient_name", "المريض")
    )
    treatment = lead_data.get("treatment", "تنظيف أسنان شامل")
    appointment_time = lead_data.get("appointment_time", "الساعة 11")

    greeting = f"السلام عليكم، معك سارة من عيادة ابتسامة الأسنان. معي يا {patient_name}؟"

    system_prompt = f"""
أنتِ سارة، منسقة علاج أسنان محترفة في عيادة ابتسامة الأسنان في عمّان.

❗ تحدثي فقط باللهجة الأردنية العمّانية.
❗ لا فصحى.
❗ لا كلمات إنجليزية.
❗ نبرة دافئة، إنسانية، طبيعية، مش روبوت.

طريقة الكلام:
- جمل قصيرة
- نفس طبيعي
- نغمة بشرية
- توقيت هادي

نطق الأسماء:
- شددي الحركات
- الاسم يُنطق بوضوح وبطء طبيعي

الهدف:
تأكيد موعد علاج أسنان وبناء ثقة.

الافتتاح (أول جملة حرفيًا):
"{greeting}"

تفاصيل الموعد:
- العلاج: {treatment}
- الوقت: {appointment_time}

لو وافق:
استدعي الدالة update_lead_status_confirmed فورًا.

لو رفض:
استدعي update_lead_status_cancelled فورًا.

احكي وكأنك إنسانة حقيقية، مش نظام.
"""

    # ===========================
    # TELNYX TRANSPORT
    # ===========================

    stream_id = "unknown"
    msg = json.loads(await websocket_client.receive_text())
    stream_id = msg.get("stream_id", "unknown")

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        inbound_encoding="PCMU",
        outbound_encoding="PCMU",
    )

    vad = _VAD_MODEL["value"]
    if vad is None:
        vad = SileroVADAnalyzer(
            params=VADParams(min_volume=0.6, stop_secs=0.3)
        )
        _VAD_MODEL["value"] = vad

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
    )

    # ===========================
    # GEMINI LIVE (FIXED)
    # ===========================

    from pipecat.services.google.gemini_live.llm import (
        GeminiLiveLLMService,
        InputParams,
    )

    gemini = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="models/gemini-live-2.5-flash-native-audio",
        system_instruction=system_prompt,
        params=InputParams(temperature=0.35),
        inference_on_context_initialization=True,
    )

    async def confirm(params: FunctionCallParams):
        await params.result_callback({"status": "confirmed"})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, 2.5))

    async def cancel(params: FunctionCallParams):
        await params.result_callback({"status": "cancelled"})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, 2.5))

    gemini.register_function("update_lead_status_confirmed", confirm)
    gemini.register_function("update_lead_status_cancelled", cancel)

    context = LLMContext(messages=[{"role": "user", "content": "ابدئي"}])

    pipeline = Pipeline(
        [
            transport.input(),
            gemini,
            transport.output(),
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()
    await runner.run(task)
