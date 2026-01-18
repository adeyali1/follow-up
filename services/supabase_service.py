import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

def create_campaign(name: str):
    data = {"name": name, "status": "active"}
    response = supabase.table("campaigns").insert(data).execute()
    return response.data[0]

def add_leads(leads: list):
    response = supabase.table("leads").insert(leads).execute()
    return response.data

def get_pending_leads(campaign_id: str):
    response = supabase.table("leads").select("*").eq("campaign_id", campaign_id).eq("call_status", "PENDING").execute()
    return response.data

def update_lead_status(lead_id: str, status: str, transcript: str = None, recording_url: str = None):
    data = {"call_status": status}
    if transcript:
        data["transcript"] = transcript
    if recording_url:
        data["recording_url"] = recording_url
        
    response = supabase.table("leads").update(data).eq("id", lead_id).execute()
    return response.data

def get_lead(lead_id: str):
    response = supabase.table("leads").select("*").eq("id", lead_id).single().execute()
    return response.data
