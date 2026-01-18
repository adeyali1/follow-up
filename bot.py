import os
import sys
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMAssistantAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMContextFrame, EndFrame
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.google.llm_vertex import GoogleVertexLLMService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from backend.services.supabase_service import update_lead_status

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot for lead: {lead_data['id']}")
    
    # 1. Services
    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.0, start_secs=0.2, stop_secs=0.4, confidence=0.5))
    
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        language="ar",
        sample_rate=8000
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

    llm = GoogleVertexLLMService(
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        project_id=os.getenv("GOOGLE_PROJECT_ID"),
        location="us-central1",
        model="gemini-1.5-flash-001",
        tools=tools
    )
    
    tts = GoogleTTSService(
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        voice_id="ar-JO-Standard-A" # Jordanian Arabic
    )

    # 3. Handlers
    async def confirm_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Confirming order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CONFIRMED')
        await result_callback("Order confirmed successfully.")

    async def cancel_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Cancelling order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CANCELLED')
        await result_callback("Order cancelled.")

    llm.register_function(
        "update_lead_status_confirmed",
        confirm_order
    )
    llm.register_function(
        "update_lead_status_cancelled",
        cancel_order
    )

    # 3. Transport (Telnyx)
    # Note: TelnyxFrameSerializer requires stream_id and call_control_id for full functionality (like hanging up)
    # In a basic setup, we might not have them immediately from the websocket handshake in the same way.
    # We will instantiate it with placeholders or extract if possible.
    # For now, we'll use a dummy stream_id if we don't have one, or rely on what Pipecat needs.
    
    # We need to know the call_control_id to hang up. 
    # If we can't get it from the websocket, we might need to pass it from the webhook handler -> main.py -> run_bot
    # But for now, let's assume we just want audio streaming working.
    
    serializer = TelnyxFrameSerializer(
        stream_id="telnyx_stream", # Placeholder
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding="PCMU", # Telnyx default
        inbound_encoding="PCMU",
        params=TelnyxFrameSerializer.InputParams(
            sample_rate=8000
        )
    )
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            session_timeout=300,
            vad_enabled=True,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000
        )
    )

    # 4. Prompt
    system_prompt = f"""You are Kawkab AI, a delivery assistant. Speak in Jordanian Arabic.
Greet {lead_data['customer_name']}. Confirm they ordered {lead_data['order_items']} for delivery at {lead_data['delivery_time']}.
If confirmed, call tool update_lead_status_confirmed().
If cancelled, call tool update_lead_status_cancelled().
If no answer or voicemail, just hang up (I will handle this via timeout or silence).
"""

    context = LLMContext(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Greet the customer now."}
        ]
    )

    # 5. Pipeline
    user_agg = LLMUserAggregator(context)
    assistant_agg = LLMAssistantAggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        assistant_agg,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    
    runner = PipelineRunner()
    
    await task.queue_frames([LLMContextFrame(context)])
    
    await runner.run(task)
