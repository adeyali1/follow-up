# Deploy Kawkab AI Backend to aaPanel VPS

This guide will walk you through deploying the Kawkab AI backend to a VPS using aaPanel. This setup ensures stable WebSocket connections for Telnyx audio streaming.

## Prerequisites

1.  **VPS**: A Virtual Private Server (Ubuntu 22.04 recommended) with **aaPanel** installed.
2.  **Domain**: A domain name (e.g., `electron.dollany.app`) pointed to your VPS IP address.
3.  **Telnyx Account**: Your Telnyx account with a phone number configured.
4.  **Google Cloud Project**: With Vertex AI enabled and a service account key JSON file.

---

## Step 1: Prepare Your Files

Ensure you have the following files ready to upload:
*   The `backend` folder (containing `main.py`, `bot.py`, `requirements.txt`, `google_key.json`, etc.).
*   The `.env` file (we will edit this on the server).

## Step 2: Upload Files to VPS

1.  Log in to your **aaPanel** dashboard.
2.  Go to **Files**.
3.  Navigate to `/www/wwwroot`.
4.  Create a new folder named `kawkab-backend`.
5.  Open `kawkab-backend` and upload:
    *   The entire `backend` folder.
    *   Your `.env` file.

**Structure should look like this:**
```text
/www/wwwroot/kawkab-backend/
├── .env
└── backend/
    ├── main.py
    ├── bot.py
    ├── requirements.txt
    ├── google_key.json
    └── ...
```

## Step 3: Configure Environment Variables

1.  Edit the `.env` file in `/www/wwwroot/kawkab-backend/.env`.
2.  **CRITICAL**: Update the `DOMAIN` variable to your actual VPS domain.

```env
# ... other keys (SUPABASE, TELNYX, etc.) ...

# UPDATE THIS TO YOUR VPS DOMAIN
DOMAIN=electron.dollany.app

# Ensure Google Project ID is correct
GOOGLE_PROJECT_ID=my-callimg-app
GOOGLE_APPLICATION_CREDENTIALS=./backend/google_key.json
```

## Step 4: Set Up Python Project in aaPanel

1.  Go to **Website** -> **Python Project** (if not available, install "Python Manager" from the App Store).
2.  Click **Add Python Project**.
3.  **Fill in the details:**
    *   **Project Name**: `kawkab-voice`
    *   **Path**: `/www/wwwroot/kawkab-backend`
    *   **Python Version**: Select **Python 3.11** (Install via Python Manager if missing).
    *   **Framework**: `Uvicorn`
    *   **Startup File**: `backend/main.py`
    *   **Run Command**: `uvicorn backend.main:app --host 0.0.0.0 --port 8765`
    *   **Port**: `8765`
4.  Check **Install Dependencies** (from `backend/requirements.txt`).
5.  Click **Confirm** / **Submit**.

aaPanel will create the virtual environment and install the required packages.

## Step 5: Configure Reverse Proxy (Map Domain to Port)

To make your app accessible via `https://electron.dollany.app`, you need a reverse proxy.

1.  Go to **Website** -> **PHP Projects** (yes, we use the standard "Add Site" for the proxy).
2.  Click **Add Site**.
3.  **Domain**: `electron.dollany.app`
4.  **PHP Version**: Pure Static (or any version, doesn't matter).
5.  Click **Submit**.
6.  Click on the newly created site name to open settings.
7.  Go to **SSL** -> **Let's Encrypt**.
    *   Select your domain.
    *   Click **Apply** to get a free SSL certificate.
    *   Enable **Force HTTPS**.
8.  Go to **Reverse Proxy**.
    *   Click **Add Reverse Proxy**.
    *   **Name**: `kawkab-api`
    *   **Target URL**: `http://127.0.0.1:8765`
    *   **Sent Domain**: `$host`
    *   Click **Submit**.

## Step 6: Verify Deployment

1.  Open your browser and visit: `https://electron.dollany.app/`
2.  You should see the message: `{"status": "running", "service": "Kawkab AI Voice Backend"}`

## Step 7: Update Telnyx

1.  The backend code is configured to automatically tell Telnyx where to send the stream for **outbound calls**.
2.  **For Inbound Calls (calling the bot)**:
    *   Go to **Telnyx Portal** -> **Phone Numbers**.
    *   Edit your number settings.
    *   **Call Control Webhook**: `https://electron.dollany.app/api/voice/webhook`
    *   Save changes.

## Troubleshooting

*   **Logs**: check the project logs in aaPanel Python Manager.
*   **Audio Issues**: Ensure ports `80` and `443` are open in the Security firewall of aaPanel and your VPS provider (AWS/DigitalOcean/etc).
