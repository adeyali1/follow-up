# Fix Missing Dependencies on aaPanel VPS

The error `ModuleNotFoundError: No module named 'telnyx'` means the required Python libraries are not installed in the virtual environment that aaPanel is using.

## Option 1: Install via aaPanel UI (Easiest)

1.  Go to **Website** -> **Python Project**.
2.  Find your project (`kawkab-voice`).
3.  Click **Module** (or "Manager").
4.  In the "Install Module" input, type each of these one by one and click **Add** (or upload your `requirements.txt` if there's an option):
    *   `telnyx`
    *   `fastapi`
    *   `uvicorn`
    *   `supabase`
    *   `pandas`
    *   `python-multipart`
    *   `python-dotenv`
    *   `pipecat-ai[deepgram,google,twilio]`
    *   `websockets`
    *   `loguru`
    *   `openpyxl`
5.  **Restart** the project.

## Option 2: Install via Terminal (More Reliable)

If the UI fails, use the terminal.

1.  **Open Terminal** in aaPanel.
2.  **Activate the Virtual Environment**:
    *   Usually located at `/www/wwwroot/kawkab-backend/kawkab-voice_venv/bin/activate` or similar.
    *   Try running:
        ```bash
        source /www/wwwroot/kawkab-backend/kawkab-voice_venv/bin/activate
        ```
        *(Note: Replace `kawkab-voice_venv` with the actual name aaPanel created. You can `ls /www/wwwroot/kawkab-backend` to find it.)*

3.  **Install Requirements**:
    ```bash
    cd /www/wwwroot/kawkab-backend
    pip install -r backend/requirements.txt
    ```

4.  **Restart the Project** in aaPanel.

## Option 3: Manual Install Command (If requirements.txt fails)

Run this single command inside the virtual environment:

```bash
pip install fastapi uvicorn supabase pandas python-multipart python-dotenv "pipecat-ai[deepgram,google,twilio]" telnyx websockets loguru openpyxl
```
