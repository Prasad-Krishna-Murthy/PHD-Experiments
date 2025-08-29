# app.py
import streamlit as st
from pathlib import Path
import os
import io
import hashlib
import json
import time
import uuid
import csv
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import librosa
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --------- NOTE ----------
# This app uses SpeechBrain's pretrained speaker embedding model.
# Install dependencies (see README below). You MUST have torch installed.
#
# pip install streamlit librosa soundfile numpy scikit-learn speechbrain
# ------------------------

# Import speechbrain model lazily (handle if not installed)
try:
    from speechbrain.pretrained import EncoderClassifier
    SB_AVAILABLE = True
except Exception as e:
    SB_AVAILABLE = False

# ---------- Config ----------
DATA_DIR = Path("voice_auth_data")
USERS_FILE = DATA_DIR / "users.pkl"        # stores user dict
LOG_FILE = DATA_DIR / "auth_logs.csv"      # log of enroll/verify attempts
UPLOAD_DIR = DATA_DIR / "uploads"          # raw uploaded clips saved here
THRESHOLD_DEFAULT = 0.75                   # cosine similarity threshold for positive match
SAMPLE_RATE = 16000                        # model/sample rate expected

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------- Utilities ----------
def hash_password(password: str, salt: bytes = None) -> Dict[str, str]:
    """Return dict with salt (hex) and hash (hex) using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return {"salt": salt.hex(), "hash": hashed.hex()}

def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hashed.hex() == hash_hex

def load_users() -> Dict[str, Any]:
    if USERS_FILE.exists():
        with open(USERS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_users(users: Dict[str, Any]):
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)

def log_event(event_type: str, username: str, details: Dict[str, Any]):
    header = ["timestamp", "event_type", "username", "details_json"]
    row = [datetime.utcnow().isoformat() + "Z", event_type, username, json.dumps(details)]
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

# ---------- Model wrapper ----------
@st.cache_resource(show_spinner=False)
def get_speechbrain_model():
    if not SB_AVAILABLE:
        raise RuntimeError("speechbrain is not installed or failed to import.")
    # Pretrained ECAPA-TDNN trained on VoxCeleb for speaker embeddings
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    return model

def extract_embedding_from_audio_bytes(wave_bytes: bytes, sr_target: int = SAMPLE_RATE) -> np.ndarray:
    """
    Loads audio bytes, resamples to sr_target, returns a normalized embedding (1D numpy).
    """
    # load with soundfile via BytesIO or librosa
    with io.BytesIO(wave_bytes) as bio:
        # soundfile supports reading file-like objects
        data, sr = sf.read(bio, dtype="float32")
        # If stereo, convert to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # if sampling rate differs, resample with librosa
        if sr != sr_target:
            data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

    # Use SpeechBrain model to get embedding
    model = get_speechbrain_model()
    # SpeechBrain expects a tensor or numpy of shape (channels, time) or (time,)
    # The classifier.encode_batch accepts waveform in (batch, time)
    # We'll pass waveform as numpy 1D inside list
    try:
        embeddings = model.encode_batch(np.expand_dims(data, axis=0))
        # embeddings is a torch tensor; convert to numpy
        emb = embeddings.squeeze().cpu().numpy()
    except Exception as e:
        # fallback: try raw waveform feed through brain's classifier
        emb = model.encode_batch(np.expand_dims(data, axis=0)).squeeze().cpu().numpy()
    # normalize (L2)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

# ---------- Enrollment & Authentication logic ----------
def enroll_user(username: str, password: str, audio_bytes_list: List[bytes]) -> Dict[str, Any]:
    """
    audio_bytes_list must contain exactly 3 clips (or we can accept >=3 and use first 3).
    Returns a dict with success flag and message.
    """
    users = load_users()
    if username in users:
        return {"success": False, "message": "User already exists."}
    if len(audio_bytes_list) < 3:
        return {"success": False, "message": "Please upload exactly 3 audio clips for enrollment."}

    # Hash password
    ph = hash_password(password)
    embeddings = []
    clip_meta = []
    for i, b in enumerate(audio_bytes_list[:3]):
        # store raw file
        filename = UPLOAD_DIR / f"{username}_enroll_{int(time.time())}_{i}.wav"
        with open(filename, "wb") as f:
            f.write(b)
        clip_meta.append({"path": str(filename), "length_bytes": len(b)})
        try:
            emb = extract_embedding_from_audio_bytes(b)
        except Exception as e:
            return {"success": False, "message": f"Failed to extract embedding from clip {i}: {e}"}
        embeddings.append(emb)

    # Average embeddings to create representative embedding
    rep_emb = np.mean(np.vstack(embeddings), axis=0)
    norm = np.linalg.norm(rep_emb)
    if norm > 0:
        rep_emb = rep_emb / norm

    # Save user record
    users[username] = {
        "password_hash": ph["hash"],
        "password_salt": ph["salt"],
        "embedding": rep_emb.astype(np.float32),  # store numpy
        "enrolled_at": datetime.utcnow().isoformat() + "Z",
        "clips": clip_meta
    }
    save_users(users)
    log_event("enroll", username, {"status": "success", "clips": clip_meta})
    return {"success": True, "message": "User enrolled successfully."}

def authenticate_user(username: str, password: str, audio_bytes: bytes, threshold: float) -> Dict[str, Any]:
    users = load_users()
    if username not in users:
        log_event("verify", username, {"status": "fail", "reason": "no_such_user"})
        return {"success": False, "message": "User not found."}
    rec = users[username]
    if not verify_password(password, rec["password_salt"], rec["password_hash"]):
        log_event("verify", username, {"status": "fail", "reason": "bad_password"})
        return {"success": False, "message": "Incorrect password."}

    # extract embedding for probe
    try:
        probe_emb = extract_embedding_from_audio_bytes(audio_bytes)
    except Exception as e:
        log_event("verify", username, {"status": "fail", "reason": f"feature_extract_error: {e}"})
        return {"success": False, "message": f"Failed to extract embedding: {e}"}

    # compute cosine similarity
    stored_emb = rec["embedding"]
    # ensure shapes
    stored_emb = np.array(stored_emb).reshape(1, -1)
    probe_emb = probe_emb.reshape(1, -1)
    sim = float(cosine_similarity(stored_emb, probe_emb)[0,0])

    details = {"similarity": sim, "threshold": threshold}
    if sim >= threshold:
        log_event("verify", username, {"status":"success", **details})
        return {"success": True, "message": f"Authentication SUCCESS (similarity={sim:.4f})", "similarity": sim}
    else:
        log_event("verify", username, {"status":"fail", **details})
        return {"success": False, "message": f"Authentication FAILED (similarity={sim:.4f})", "similarity": sim}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Voice Authentication (ECAPA embeddings)", layout="wide")

st.title("Voice Authentication System — Enrollment & Validation")
st.markdown(
    """
    This demo enrolls users with password + 3 audio clips, extracts speaker embeddings (ECAPA-TDNN via SpeechBrain),
    stores an average embedding per user, and then validates any uploaded clip against the stored embedding using cosine similarity.
    """
)

if not SB_AVAILABLE:
    st.error("The `speechbrain` package is not available. Install dependencies: `pip install speechbrain` and ensure `torch` is installed.")
    st.stop()

# Sidebar: show users & logs
with st.sidebar:
    st.header("Settings & Data")
    threshold = st.slider("Authentication similarity threshold", 0.50, 0.95, THRESHOLD_DEFAULT, step=0.01)
    st.write("Data directory:", str(DATA_DIR.resolve()))
    st.write("Registered users:", len(load_users()))
    if st.button("Show last 20 log entries"):
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding='utf-8') as f:
                lines = f.readlines()[-21:]  # header + last 20
            st.text("".join(lines[-21:]))
        else:
            st.info("No log file yet.")

tabs = st.tabs(["Enroll user", "Validate speaker", "Admin: View users & logs"])

# ---------- Tab 1: Enrollment ----------
with tabs[0]:
    st.header("1) Enroll user (requires 3 audio clips)")
    with st.form("enroll_form"):
        username = st.text_input("Username (unique)")
        password = st.text_input("Password", type="password")
        st.markdown("Upload exactly **3** audio files (wav/mp3). Each should be at least 2 seconds if possible.")
        uploaded = st.file_uploader("Upload 3 clips", type=["wav","mp3","flac","ogg"], accept_multiple_files=True)
        submit = st.form_submit_button("Enroll user")

    if submit:
        if not username or not password:
            st.warning("Provide username and password.")
        else:
            if uploaded is None or len(uploaded) < 3:
                st.warning("Please upload at least 3 audio files.")
            else:
                # read bytes
                audio_bytes_list = []
                for f in uploaded[:3]:
                    audio_bytes_list.append(f.read())
                with st.spinner("Extracting embeddings and enrolling... this may take a few seconds"):
                    result = enroll_user(username.strip(), password, audio_bytes_list)
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])

# ---------- Tab 2: Validation ----------
with tabs[1]:
    st.header("2) Validate (test) speaker")
    with st.form("verify_form"):
        v_username = st.text_input("Username to verify")
        v_password = st.text_input("Password", type="password", key="vpass")
        probe_file = st.file_uploader("Upload any clip for verification (wav/mp3/flac/ogg)", type=["wav","mp3","flac","ogg"], key="probe")
        verify_btn = st.form_submit_button("Verify speaker")
    if verify_btn:
        if not v_username or not v_password or not probe_file:
            st.warning("Provide username, password and upload a probe clip.")
        else:
            probe_bytes = probe_file.read()
            with st.spinner("Extracting embedding and comparing..."):
                res = authenticate_user(v_username.strip(), v_password, probe_bytes, threshold)
            if res["success"]:
                st.success(res["message"])
            else:
                st.error(res["message"])
            st.write("Similarity score (float):", res.get("similarity"))

# ---------- Tab 3: Admin ----------
with tabs[2]:
    st.header("Admin / Debug")
    st.subheader("Registered users")
    users = load_users()
    if users:
        for uname, rec in users.items():
            st.write(f"User: **{uname}** — enrolled at {rec.get('enrolled_at')}")
            st.write("Stored embedding shape:", np.array(rec.get("embedding")).shape)
            st.write("Clips:", rec.get("clips", []))
            if st.button(f"Delete user {uname}"):
                users.pop(uname)
                save_users(users)
                st.success(f"Deleted user {uname}")
                st.experimental_rerun()
    else:
        st.info("No users registered yet.")

    st.subheader("View logs (last 200 lines)")
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding='utf-8') as f:
            content = f.read().splitlines()
            table = content[-200:]
            st.text("\n".join(table))
    else:
        st.info("No logs yet.")
