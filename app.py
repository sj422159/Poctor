from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import subprocess
import threading
import os
import signal
import atexit
import time

app = FastAPI()

active_sessions = {}
EXAM_DURATION_MINUTES = 10
WARNING_LIMIT = 3
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


class StartRequest(BaseModel):
    user_id: str


class StopRequest(BaseModel):
    user_id: str


class SubmitRequest(BaseModel):
    user_id: str


def log_event(user_id, message):
    with open(f"{LOG_DIR}/{user_id}.log", "a") as f:
        f.write(f"{datetime.now()} - {message}\n")


def monitor_timer(user_id):
    session = active_sessions.get(user_id)
    if not session:
        return
    end_time = session["start_time"] + timedelta(minutes=EXAM_DURATION_MINUTES)

    while datetime.now() < end_time:
        time.sleep(1)
        if user_id not in active_sessions:
            return  # Session stopped manually
    log_event(user_id, "Time up. Auto-submitting.")
    stop_and_submit(user_id, auto=True)


def stop_and_submit(user_id, auto=False):
    session = active_sessions.get(user_id)
    if not session:
        return
    proc = session["process"]
    try:
        proc.terminate()
        proc.wait()
    except Exception as e:
        log_event(user_id, f"Error terminating process: {e}")
    finally:
        del active_sessions[user_id]
    reason = "Auto" if auto else "Manual"
    log_event(user_id, f"{reason} submission. Session ended.")


@app.get("/")
def root():
    return {"message": "AI Proctor API is running"}


@app.post("/start-proctor")
def start_proctor(req: StartRequest):
    user_id = req.user_id
    if user_id in active_sessions:
        raise HTTPException(status_code=400, detail="Session already running.")

    try:
        proc = subprocess.Popen(["python", "main.py", user_id])
        active_sessions[user_id] = {
            "process": proc,
            "start_time": datetime.now(),
            "warnings": 0
        }
        threading.Thread(target=monitor_timer, args=(user_id,), daemon=True).start()
        log_event(user_id, "Proctoring session started.")
        return {"status": "started", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop-proctor")
def stop_proctor(req: StopRequest):
    user_id = req.user_id
    if user_id not in active_sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    stop_and_submit(user_id)
    return {"status": "stopped", "user_id": user_id}


@app.post("/submit-exam")
def manual_submit(req: SubmitRequest):
    user_id = req.user_id
    if user_id not in active_sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    stop_and_submit(user_id)
    return {"status": "submitted", "user_id": user_id}


@app.post("/add-warning")
def add_warning(req: SubmitRequest):
    user_id = req.user_id
    if user_id not in active_sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    active_sessions[user_id]["warnings"] += 1
    warning_count = active_sessions[user_id]["warnings"]
    log_event(user_id, f"Warning issued. Count: {warning_count}")
    if warning_count >= WARNING_LIMIT:
        log_event(user_id, f"Exceeded {WARNING_LIMIT} warnings. Terminating session.")
        stop_and_submit(user_id, auto=True)
    return {"warnings": warning_count}


# Cleanup on shutdown
def cleanup_sessions():
    for user_id, session in list(active_sessions.items()):
        log_event(user_id, "Server shutdown. Cleaning up session.")
        try:
            session["process"].terminate()
            session["process"].wait()
        except Exception as e:
            log_event(user_id, f"Error during cleanup: {e}")
    active_sessions.clear()


atexit.register(cleanup_sessions)
signal.signal(signal.SIGTERM, lambda s, f: cleanup_sessions())
signal.signal(signal.SIGINT, lambda s, f: cleanup_sessions())