import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import telnyx
import aiohttp
from dotenv import load_dotenv
from services import supabase_service
from bot import run_bot
import uvicorn
import asyncio
import json
import base64
from urllib.parse import quote

load_dotenv()

app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "running", "service": "Kawkab AI Voice Backend"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Telnyx Configuration
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
telnyx_client = telnyx.Telnyx(api_key=TELNYX_API_KEY)
TELNYX_CONNECTION_ID = os.getenv("TELNYX_CONNECTION_ID")
TELNYX_PHONE_NUMBER = os.getenv("TELNYX_PHONE_NUMBER")
DOMAIN = os.getenv("DOMAIN", "apt-generic-burner-abilities.trycloudflare.com")

@app.post("/api/campaigns/create")
async def create_campaign_api(request: Request):
    try:
        data = await request.json()
        name = data.get("name", "New Campaign")
        campaign = supabase_service.create_campaign(name)
        return {"id": campaign['id'], "name": campaign['name']}
    except Exception as e:
        print(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/campaigns/manual")
async def create_manual_campaign(request: Request):
    try:
        data = await request.json()
        campaign_name = data.get("name", "Manual Campaign")
        manual_leads = data.get("leads", [])
        
        # Create Campaign
        campaign = supabase_service.create_campaign(campaign_name)
        
        # Prepare Leads
        leads = []
        for row in manual_leads:
            phone = str(row.get("phone", ""))
            name = row.get("name", "Customer")
            items = row.get("items", "")
            time = str(row.get("time", ""))

            # Clean phone number
            if phone.endswith('.0'):
                phone = phone[:-2]
            if phone and not phone.startswith('+'):
                phone = '+' + phone
                
            leads.append({
                "campaign_id": campaign['id'],
                "customer_name": name,
                "phone_number": phone,
                "order_items": items,
                "delivery_time": time,
                "call_status": "PENDING"
            })
        
        # Insert Leads
        if leads:
            supabase_service.add_leads(leads)
            
        return {"campaign_id": campaign['id'], "leads_count": len(leads)}
    except Exception as e:
        print(f"Error creating manual campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-leads")
async def upload_leads(file: UploadFile = File(...), campaign_name: str = "New Campaign"):
    content = await file.read()
    
    # Parse file
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(content))
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Create Campaign
    campaign = supabase_service.create_campaign(campaign_name)
    
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Columns found: {df.columns}")

    # Prepare Leads
    leads = []
    for _, row in df.iterrows():
        # Map common variations
        phone = str(row.get("phone", row.get("mobile", row.get("number", ""))))
        name = row.get("name", row.get("customer", "Customer"))
        items = row.get("items", row.get("order", ""))
        time = str(row.get("time", row.get("delivery_time", "")))

        # Clean phone number
        if phone.endswith('.0'):
            phone = phone[:-2]
        if phone and not phone.startswith('+'):
            phone = '+' + phone
            
        leads.append({
            "campaign_id": campaign['id'],
            "customer_name": name,
            "phone_number": phone,
            "order_items": items,
            "delivery_time": time,
            "call_status": "PENDING"
        })
    
    # Insert Leads
    if leads:
        supabase_service.add_leads(leads)
        
    return {"campaign_id": campaign['id'], "leads_count": len(leads)}

async def process_campaign(campaign_id: str):
    leads = supabase_service.get_pending_leads(campaign_id)
    print(f"Starting campaign {campaign_id} with {len(leads)} leads")
    
    for lead in leads:
        try:
            # Initiate Call
            print(f"Processing lead: {lead}")
            print(f"Calling {lead.get('phone_number')}...")
            
            if not lead.get('phone_number'):
                print(f"Skipping lead {lead['id']} due to missing phone number")
                continue

            # Determine the webhook URL
            webhook_url = f"https://{DOMAIN}/api/voice/webhook?lead_id={lead['id']}"
            
            # Encode Client State with Lead ID
            client_state = base64.b64encode(json.dumps({'lead_id': lead['id']}).encode()).decode()

            call = telnyx_client.calls.dial(
                to=lead['phone_number'],
                from_=TELNYX_PHONE_NUMBER,
                connection_id=TELNYX_CONNECTION_ID,
                webhook_url=webhook_url,
                client_state=client_state
            )
            # Handle response object safely
            call_control_id = getattr(call, 'call_control_id', None)
            if not call_control_id and hasattr(call, 'data'):
                 call_control_id = getattr(call.data, 'call_control_id', None)
            
            print(f"Call initiated. ID: {call_control_id}")
            print(f"Raw Call Response: {call}")

            supabase_service.update_lead_status(lead['id'], "CALLED")
            # Sleep a bit to avoid rate limits
            await asyncio.sleep(1) 
        except Exception as e:
            print(f"Error calling {lead['id']}: {e}")

@app.post("/start-campaign/{campaign_id}")
async def start_campaign(campaign_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_campaign, campaign_id)
    return {"status": "Campaign started"}

@app.post("/api/voice/webhook")
async def webhook_route(request: Request):
    # Handle Telnyx Webhook
    try:
        body = await request.json()
        data = body.get('data', {})
        event_type = data.get('event_type')
        payload = data.get('payload', {})
        call_control_id = payload.get('call_control_id')
        
        # Extract lead_id from client_state if available, or query params
        client_state_str = payload.get('client_state')
        lead_id = None
        if client_state_str:
            try:
                client_state = json.loads(base64.b64decode(client_state_str).decode())
                lead_id = client_state.get('lead_id')
            except:
                pass
        
        if not lead_id:
             lead_id = request.query_params.get("lead_id")

        print(f"Received Telnyx event: {event_type} for Lead: {lead_id}")

        if event_type == 'call.answered':
            # Fork Audio to WebSocket
            # Clean domain to avoid double slashes
            clean_domain = DOMAIN.replace("https://", "").replace("http://", "").strip("/")
            stream_url = f"wss://{clean_domain}/media-stream?lead_id={lead_id}&call_control_id={call_control_id}"
            print(f"Starting audio stream to: {stream_url}")
            
            # Use Telnyx SDK (Client based)
            try:
                # SDK expects the raw ID
                print(f"Using Raw ID for SDK: {call_control_id}")
                
                telnyx_client.calls.actions.start_streaming(
                    call_control_id,
                    stream_url=stream_url,
                    stream_track="both_tracks"
                )
                print(f"Start streaming success (SDK) for {call_control_id}")
            except Exception as e:
                print(f"Failed to start streaming (SDK): {e}")
                
                # Fallback to aiohttp with ENCODED ID and CORRECT ENDPOINT
                try:
                    encoded_call_control_id = quote(call_control_id, safe='')
                    print(f"Fallback: Using Encoded ID: {encoded_call_control_id}")
                    
                    async with aiohttp.ClientSession() as session:
                        # Correct endpoint is streaming_start, not fork_media
                        fork_url = f"https://api.telnyx.com/v2/calls/{encoded_call_control_id}/actions/streaming_start"
                        headers = {
                            "Authorization": f"Bearer {TELNYX_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        payload_data = {
                            "stream_url": stream_url,
                            "stream_track": "both_tracks"
                        }
                        async with session.post(fork_url, headers=headers, json=payload_data) as response:
                             print(f"Fallback HTTP Status: {response.status}")
                             resp_text = await response.text()
                             print(f"Fallback HTTP Response: {resp_text}")
                except Exception as ex:
                    print(f"Fallback failed: {ex}")
            
        elif event_type == 'call.hangup':
            print(f"Call ended for lead {lead_id}")
            
        return JSONResponse({"status": "ok"})
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    
    # Get lead_id from query
    lead_id = websocket.query_params.get("lead_id")
    call_control_id = websocket.query_params.get("call_control_id")

    if not lead_id:
        print("No lead_id provided in WebSocket connection")
        await websocket.close()
        return

    lead = supabase_service.get_lead(lead_id)
    if not lead:
        print(f"Lead not found for id: {lead_id}")
        await websocket.close()
        return

    print(f"WebSocket connected for lead {lead_id}, call_control_id: {call_control_id}")

    # Telnyx sends an initial message with metadata (sometimes)
    # We might receive a 'connected' event first
    # We need to get the call_control_id if we want to hang up from the bot
    # Usually passed in the metadata or we can infer it if we tracked it
    
    # For now, we will pass the connection to the bot
    # The bot will handle the stream
    
    await run_bot(websocket, lead, call_control_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
