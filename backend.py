# backend_app.py
from fastapi import FastAPI, HTTPException, Request, Query, UploadFile, File, Depends, status, Response # Added Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
import time
from datetime import datetime, timedelta
from transformers import pipeline
import torch
from io import StringIO
import pandas as pd
import os
import logging
import json
from bcrypt import hashpw, gensalt, checkpw
from typing import Optional, List, Dict, Any
import textstat
import re
import mysql.connector
from mysql.connector import Error
import logging
import os
logging.basicConfig(level=logging.INFO)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 3306) # Added default port for robustness
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Sivamani@2112") # !!! MAKE SURE THIS IS CORRECT !!!
DB_NAME = os.getenv("DB_NAME", "milestone2_new")
db = None
cursor = None
def get_db_connection():
    global db, cursor
    if db is not None and db.is_connected():
        return db, cursor
    try:
        db = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            auth_plugin='mysql_native_password' 
        )
        cursor = db.cursor(dictionary=True)
        return db, cursor
    except Error as e:
        logging.error(f"❌ MySQL connection failed: {e}")
        db = None
        cursor = None
        return None, None
# --- Evaluation Metric (ROUGE) ---
from rouge_score import rouge_scorer
# backend.py (Place near the top, after imports and config)

DB_FILE = "db.json"

def load_db():
    """Loads the database from db.json or initializes it."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading DB from file: {e}. Initializing empty DB.")
            return {
                "users": {},
                "history": {},
                "logins": [],
                "feedback": [],
                "analytics": {"unique_users": 0, "total_generated_texts": 0, "model_usage_counts": {}}
            }
    else:
        logging.info("DB file not found. Initializing empty DB.")
        return {
            "users": {},
            "history": {},
            "logins": [],
            "feedback": [],
            "analytics": {"unique_users": 0, "total_generated_texts": 0, "model_usage_counts": {}}
        }

def save_db():
    """Saves the database to db.json."""
    try:
        with open(DB_FILE, "w") as f:
            json.dump(DB, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving DB to file: {e}")

# Global database dictionary
DB = load_db()

# --- Application Initialization (FastAPI) ---
app = FastAPI() 
# ... rest of your app setup
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# Environment & Model Config
# --------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_key_that_should_be_long_and_random")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
RESET_TOKEN_EXPIRE_MINUTES = 10

TRANSFORMERS_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE_DIR", "./models_cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")
logging.info(f"Model cache directory: {TRANSFORMERS_CACHE_DIR}")

SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
PARAPHRASING_MODEL_NAME = "google/pegasus-xsum"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")

# FIX 1: Removed redundant 'english' model. The `translate_text` function handles English by returning the original text.
TRANSLATION_MODELS = {
    "hindi": "Helsinki-NLP/opus-mt-en-hi",
    "telugu": "Meher2006/english-to-telugu-model"
}

nlp_pipelines = {}

DATASET_MAPPING: Dict[str, str] = {}
DATASET_FILE = "evaluation_data.csv"
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "./finetuned_t5_summarizer")

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.casefold()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_dataset_mapping():
    global DATASET_MAPPING
    logging.info(f"Attempting to load dataset from {DATASET_FILE} for reference lookup.")
    if not os.path.exists(DATASET_FILE):
        logging.warning(f"Dataset file '{DATASET_FILE}' not found. Reference text lookup will be disabled.")
        return
    try:
        df = pd.read_csv(DATASET_FILE)
        if 'text' in df.columns and 'reference_text' in df.columns:
            df['text'] = df['text'].astype(str).apply(normalize_text)
            DATASET_MAPPING = df.set_index('text')['reference_text'].to_dict()
            logging.info(f"Loaded {len(DATASET_MAPPING)} reference entries.")
        else:
            logging.error("Dataset CSV must contain 'text' and 'reference_text' columns.")
    except Exception as e:
        logging.error(f"Error loading dataset mapping: {e}")

# --------------------------
# Mock Database with File Persistence
# --------------------------
MOCK_DB_FILE = "user_data.json"
MOCK_USERS_DB: Dict[str, Any] = {}
RECENT_LOGINS: List[Dict[str, str]] = []  # {'user_email', 'timestamp'}

def load_user_data():
    global MOCK_USERS_DB
    if os.path.exists(MOCK_DB_FILE):
        try:
            with open(MOCK_DB_FILE, 'r') as f:
                MOCK_USERS_DB = json.load(f)
                for email, data in MOCK_USERS_DB.items():
                    if 'role' not in data:
                        data['role'] = 'admin' if email.lower() == ADMIN_EMAIL.lower() else 'user'
            logging.info(f"Loaded {len(MOCK_USERS_DB)} user records from {MOCK_DB_FILE}")
        except Exception as e:
            logging.error(f"Error loading mock DB: {e}. Starting with empty database.")
            MOCK_USERS_DB = {}
    else:
        logging.info(f"Mock DB file {MOCK_DB_FILE} not found. Starting with empty database.")

def save_user_data():
    try:
        with open(MOCK_DB_FILE, 'w') as f:
            json.dump(MOCK_USERS_DB, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving mock DB: {e}")

def load_pipelines():
    """Load models but fail gracefully if unavailable (so API can still serve non-model endpoints)."""
    global nlp_pipelines
    try:
        logging.info(f"Loading Summarization (BART) model: {SUMMARIZATION_MODEL_NAME}")
        nlp_pipelines['summarization'] = pipeline(
            "summarization",
            model=SUMMARIZATION_MODEL_NAME,
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        logging.error(f"Failed to load summarization model: {e}")

    try:
        logging.info(f"Loading Paraphrasing (PEGASUS) model: {PARAPHRASING_MODEL_NAME}")
        # FIX 2: Use 'text2text-generation' as the task for the PEGASUS (paraphrasing) model
        nlp_pipelines['paraphrasing'] = pipeline(
            "text2text-generation",
            model=PARAPHRASING_MODEL_NAME,
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        logging.error(f"Failed to load paraphrasing model: {e}")

    for lang, model_name in TRANSLATION_MODELS.items():
        try:
            logging.info(f"Loading translation model for {lang}: {model_name}")
            
            # FIX 3: Assign task dynamically based on model name for higher success rate
            if "opus-mt" in model_name:
                # Use standard Opus-MT translation task naming convention
                task = f"translation_en_to_{lang.lower()[:2]}"
            else:
                # Use generic text2text-generation for models like t5-small or Meher2006
                task = "text2text-generation"

            nlp_pipelines[f'translate_{lang}'] = pipeline(
                task,
                model=model_name,
                device=0 if DEVICE == "cuda" else -1
            )
        except Exception as e:
            logging.error(f"Failed to load translation model for {lang} ({model_name}): {e}")

    load_dataset_mapping()
    logging.info("Model loading attempted (some models may be unavailable).")

# Initialization
load_user_data()
load_pipelines()

# --------------------------
# Auth Helpers
# --------------------------
def hash_password(password: str) -> str:
    return hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')

def check_password(password: str, hashed_password: str) -> bool:
    return checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    user_email = data.get("sub")
    if user_email and user_email in MOCK_USERS_DB:
        to_encode["role"] = MOCK_USERS_DB[user_email].get('role', 'user')
    else:
        to_encode["role"] = 'user'
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_reset_token(email: str):
    expire = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": email, "exp": expire, "type": "reset"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_reset_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "reset":
            raise jwt.InvalidTokenError("Incorrect token type.")
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Reset token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid reset token.")
    except Exception as e:
        logging.error(f"Reset token decode error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed during reset.")

def authenticate_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in MOCK_USERS_DB:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload or user not found")
        payload['role'] = MOCK_USERS_DB[email].get('role', 'user')
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except Exception as e:
        logging.error(f"Auth error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")

def authenticate_admin(auth_payload: dict = Depends(authenticate_user)):
    email = auth_payload["sub"]
    user_role = MOCK_USERS_DB.get(email, {}).get('role', 'user')
    if user_role != 'admin':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied. Admin privileges required.")
    return auth_payload

def log_history(email: str, task: str, model: str, original_text: str, result: dict):
    if email not in MOCK_USERS_DB:
        MOCK_USERS_DB[email] = {'history': [], 'hashed_password': '', 'role': 'user'}
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "model": model,
        "original_text_snippet": (original_text[:100] + "...") if original_text else "",
        "feedback_score": None,
        "feedback_comment": None,
        **(result if isinstance(result, dict) else {})
    }
    MOCK_USERS_DB[email].setdefault('history', []).append(log_entry)
    save_user_data()

def get_user_history(email: str):
    return MOCK_USERS_DB.get(email, {}).get('history', [])

def update_history_feedback(email: str, timestamp: str, score: int, comment: Optional[str]):
    if email not in MOCK_USERS_DB or 'history' not in MOCK_USERS_DB[email]:
        logging.warning(f"User {email} not found or no history.")
        return False
    history = MOCK_USERS_DB[email]['history']
    for entry in history:
        if entry.get('timestamp') == timestamp:
            entry['feedback_score'] = score
            entry['feedback_comment'] = comment
            save_user_data()
            logging.info(f"Feedback logged for {email} at {timestamp}.")
            return True
    logging.warning(f"History entry with timestamp {timestamp} not found for user {email}.")
    return False
# ... (after update_history_feedback function)

def find_history_entry(email: str, timestamp: str) -> Optional[Dict[str, Any]]:
    """Finds a single history entry by email and timestamp."""
    if email not in MOCK_USERS_DB or 'history' not in MOCK_USERS_DB[email]:
        return None
    for entry in MOCK_USERS_DB[email]['history']:
        if entry.get('timestamp') == timestamp:
            return entry
    return None

def delete_history_entry(email: str, timestamp: str) -> bool:
    """Deletes a history entry by email and timestamp."""
    if email not in MOCK_USERS_DB or 'history' not in MOCK_USERS_DB[email]:
        return False
    
    history = MOCK_USERS_DB[email]['history']
    original_len = len(history)
    
    # Filter out the entry with the matching timestamp
    MOCK_USERS_DB[email]['history'] = [
        entry for entry in history if entry.get('timestamp') != timestamp
    ]
    
    new_len = len(MOCK_USERS_DB[email]['history'])
    
    if new_len < original_len:
        save_user_data()
        logging.info(f"History entry deleted for {email} at {timestamp}.")
        return True
    logging.warning(f"History entry with timestamp {timestamp} not found for user {email} during deletion.")
    return False

def update_history_entry(email: str, timestamp: str, new_output: str) -> bool:
    """Updates the generated output of a history entry and recalculates metrics."""
    entry = find_history_entry(email, timestamp)
    if entry:
        # 1. Update the content
        entry['generated_output'] = new_output
        
        # 2. Recalculate metrics based on new output
        reference_text = entry.get('reference_text')
        entry['rouge_metrics'] = calculate_rouge(new_output, reference_text)
        entry['readability_metrics'] = calculate_readability(new_output)
        
        # Recalculate spider chart data (relies on metrics being recalculated)
        entry['spider_chart_data'] = prepare_spider_chart_data(
            entry['rouge_metrics'], 
            entry['readability_metrics']
        )
        
        # Clear/flag translated output as stale
        # The translation model requires knowing the target language, which isn't saved in log_history.
        entry['translated_output'] = f"Updated content pending re-translation." 

        save_user_data()
        logging.info(f"History entry updated for {email} at {timestamp}.")
        return True
    logging.warning(f"History entry with timestamp {timestamp} not found for user {email} during update.")
    return False
def list_all_history():
    all_data = []
    for email, user_data in MOCK_USERS_DB.items():
        for entry in user_data.get('history', []):
            entry_copy = entry.copy()
            entry_copy['user_email'] = email
            all_data.append(entry_copy)
    all_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return all_data

# --------------------------
# App & CORS
# --------------------------
app = FastAPI(title="NLP Multi-Task API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Schemas
# --------------------------
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_role: str

class TextLookup(BaseModel):
    text: str
    task: str

class TaskRequest(BaseModel):
    text: str
    model: str
    language: str
    task: str
    reference_text: Optional[str] = None
    min_length: int = 30
    max_length: int = 150

class SpiderChartMetric(BaseModel):
    name: str
    value: float
    category: str
    max_value: float

class TaskResult(BaseModel):
    original_text: str
    generated_output: str
    reference_text: Optional[str]
    translated_output: str
    rouge_metrics: Dict[str, Any]
    readability_metrics: Dict[str, Any]
    elapsed_time: float
    model_name: str
    task: str
    reference_status: str
    spider_chart_data: List[SpiderChartMetric]
    timestamp: Optional[str] = None
class ContentUpdate(BaseModel):
    user_email: str
    generated_output: str
class FeedbackRequest(BaseModel):
    timestamp: str
    score: int
    comment: Optional[str] = None

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

# --------------------------
# Evaluation & Translation Helpers
# --------------------------
def calculate_rouge(generated: str, reference: Optional[str]):
    if not reference:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return {
            'rouge-1': round(scores['rouge1'].fmeasure, 4),
            'rouge-2': round(scores['rouge2'].fmeasure, 4),
            'rouge-l': round(scores['rougeL'].fmeasure, 4)
        }
    except Exception as e:
        logging.error(f"ROUGE calculation error: {e}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

def calculate_readability(text: str) -> Dict[str, float]:
    if not text.strip():
        return {"flesch_reading_ease": 0.0, "smog_index": 0.0,
                "automated_readability_index": 0.0, "coleman_liau_index": 0.0}
    try:
        return {
            "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
            "smog_index": round(textstat.smog_index(text), 2),
            "automated_readability_index": round(textstat.automated_readability_index(text), 2),
            "coleman_liau_index": round(textstat.coleman_liau_index(text), 2)
        }
    except Exception as e:
        logging.error(f"Readability error: {e}")
        return {"flesch_reading_ease": 0.0, "smog_index": 0.0,
                "automated_readability_index": 0.0, "coleman_liau_index": 0.0}

def translate_text(text: str, target_lang: str) -> str:
    target_lang_lower = target_lang.lower()

    # Handle English case separately
    if target_lang_lower == "english":
        return text

    pipeline_key = f'translate_{target_lang_lower}'
    if target_lang_lower not in TRANSLATION_MODELS:
        return f"Translation model for '{target_lang}' not found."
    try:
        translator = nlp_pipelines.get(pipeline_key)
        if not translator:
            return f"Translation model for '{target_lang}' not loaded."
        res = translator(text, max_length=400)
        # Many transformers return different keys; try common ones
        first = res[0]
        return first.get('translation_text') or first.get('generated_text') or first.get('summary_text') or str(first)
    except Exception as e:
        logging.error(f"Translation failed for {target_lang}: {e}")
        return f"Translation error: {e}"

def prepare_spider_chart_data(rouge_metrics: Dict[str, Any], readability_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    chart_data = []
    READABILITY_MAX_GRADE = 20.0
    READABILITY_MAX_FLESCH = 100.0
    ROUGE_MAX = 1.0
    for key, value in rouge_metrics.items():
        chart_data.append({
            "name": key.upper(),
            "value": float(value),
            "category": "Evaluation (Higher is better)",
            "max_value": ROUGE_MAX
        })
    chart_data.append({
        "name": "Flesch Reading Ease",
        "value": float(readability_metrics.get("flesch_reading_ease", 0.0)),
        "category": "Readability (Higher is better)",
        "max_value": READABILITY_MAX_FLESCH
    })
    chart_data.append({
        "name": "SMOG Index (Grade Level)",
        "value": float(readability_metrics.get("smog_index", 0.0)),
        "category": "Readability (Lower Grade is easier)",
        "max_value": READABILITY_MAX_GRADE
    })
    chart_data.append({
        "name": "ARI (Grade Level)",
        "value": float(readability_metrics.get("automated_readability_index", 0.0)),
        "category": "Readability (Lower Grade is easier)",
        "max_value": READABILITY_MAX_GRADE
    })
    chart_data.append({
        "name": "Coleman Liau (Grade Level)",
        "value": float(readability_metrics.get("coleman_liau_index", 0.0)),
        "category": "Readability (Lower Grade is easier)",
        "max_value": READABILITY_MAX_GRADE
    })
    return chart_data

def chunk_text(text: str, max_words: int = 400) -> list:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def safe_generate(pipe, chunk: str, min_len: int, max_len: int, chunk_index: int) -> str:
    try:
        out = pipe(chunk, min_length=min_len, max_length=max_len, truncation=True, do_sample=False)[0]
        return out.get("summary_text") or out.get("generated_text") or out.get("summary") or chunk
    except Exception as inner_e:
        logging.warning(f"Chunk {chunk_index} failed once: {inner_e}. Retrying...")
        try:
            out = pipe(chunk, max_length=max_len, truncation=True, do_sample=False)[0]
            return out.get("summary_text") or out.get("generated_text") or out.get("summary") or chunk
        except Exception as retry_e:
            logging.error(f"Chunk {chunk_index} failed again: {retry_e}. Using original chunk.")
            return chunk

# --------------------------
# Auth Endpoints
# --------------------------
@app.post("/register")
def register(user: UserCreate):
    if user.email in MOCK_USERS_DB:
        raise HTTPException(status_code=400, detail="User already exists")
    role = 'admin' if user.email.lower() == ADMIN_EMAIL.lower() else 'user'
    hashed_password = hash_password(user.password)
    MOCK_USERS_DB[user.email] = {"hashed_password": hashed_password, "history": [], "role": role}
    save_user_data()
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return {"message": "Registration successful", "access_token": access_token, "token_type": "bearer", "user_role": role}

@app.post("/login", response_model=Token)
def login_for_access_token(user_data: UserCreate):
    user_record = MOCK_USERS_DB.get(user_data.email)
    if not user_record or not check_password(user_data.password, user_record["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role = user_record.get('role', 'user')
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user_data.email}, expires_delta=access_token_expires)
    # Log recent login
    now_ts = datetime.now().isoformat()
    RECENT_LOGINS.insert(0, {"user_email": user_data.email, "timestamp": now_ts})
    # keep only last 50
    if len(RECENT_LOGINS) > 50:
        RECENT_LOGINS.pop()
    # update last_login in user record
    MOCK_USERS_DB[user_data.email]['last_login'] = now_ts
    save_user_data()
    return {"access_token": access_token, "token_type": "bearer", "user_role": role}

@app.post("/forgot_password")
def forgot_password(request: PasswordResetRequest):
    email = request.email
    if email not in MOCK_USERS_DB:
        logging.warning(f"Attempted password reset for unknown user: {email}")
        return {"message": "If the email is registered, a password reset link has been sent."}
    reset_token = create_reset_token(email)
    # Mock reset link: frontend expects token query param
    reset_url = f"http://localhost:8501/?token={reset_token}"
    logging.info(f"MOCK PASSWORD RESET LINK for {email}: {reset_url}")
    return {"message": "Password reset link sent (check server logs for the mock link)."}

@app.post("/reset_password")
def reset_password(request: PasswordResetConfirm):
    try:
        email = decode_reset_token(request.token)
    except HTTPException as e:
        raise e
    if email not in MOCK_USERS_DB:
        raise HTTPException(status_code=404, detail="User not found.")
    new_hashed_password = hash_password(request.new_password)
    MOCK_USERS_DB[email]["hashed_password"] = new_hashed_password
    save_user_data()
    logging.info(f"Password successfully reset for user: {email}")
    return {"message": "Password updated successfully. You can now log in."}

@app.get("/validate_token")
def validate_token(auth_payload: dict = Depends(authenticate_user)):
    return {"email": auth_payload["sub"], "role": auth_payload["role"]}

# --------------------------
# Dataset Lookup Endpoint
# --------------------------
@app.post("/get_reference_text")
def get_reference_text(request: TextLookup):
    input_text = normalize_text(request.text)
    reference = DATASET_MAPPING.get(input_text)
    if reference:
        return {"reference_text": reference, "status": "found"}
    else:
        return {"reference_text": None, "status": "not_found"}

# --------------------------
# NLP Task Endpoint
# --------------------------
@app.post("/process_task", response_model=TaskResult)
def process_task(request: TaskRequest, auth_payload: dict = Depends(authenticate_user)):
    email = auth_payload["sub"]
    task_type = "summarization" if request.task.lower() == "summarization" else "paraphrasing"
    if task_type not in nlp_pipelines:
        raise HTTPException(status_code=503, detail=f"Model for {task_type} is not loaded.")
    pipe = nlp_pipelines[task_type]

    t0 = time.time()
    try:
        min_len = max(10, request.min_length or 30)
        max_len = request.max_length or 200
        if max_len <= min_len:
            max_len = min_len + 50

        text_for_generation = request.text.strip()
        if len(text_for_generation.split()) > 3000:
            logging.warning("Input text too long, truncating to 3000 words")
            text_for_generation = " ".join(text_for_generation.split()[:3000])

        text_for_lookup = normalize_text(request.text)

        chunks = chunk_text(text_for_generation, max_words=400)
        generated_chunks = [safe_generate(pipe, chunk, min_len, max_len, i) for i, chunk in enumerate(chunks)]
        generated = " ".join(generated_chunks).strip()
    except Exception as e:
        logging.error(f"NLP generation failed for {task_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    manual_ref = request.reference_text.strip() if request.reference_text else None
    reference_text = None
    if manual_ref:
        reference_text = manual_ref
        reference_status = "manually_provided"
    else:
        dataset_ref = DATASET_MAPPING.get(text_for_lookup, None)
        if dataset_ref:
            reference_text = dataset_ref
            reference_status = "found_in_dataset"
        else:
            reference_text = None
            reference_status = "not_found"

    rouge_metrics = calculate_rouge(generated, reference_text)
    translated = translate_text(generated, request.language)
    readability_metrics = calculate_readability(generated)
    spider_chart_data = prepare_spider_chart_data(rouge_metrics, readability_metrics)

    t1 = time.time()
    elapsed_time = t1 - t0
    current_timestamp = datetime.now().isoformat()

    result = {
        "original_text": request.text,
        "generated_output": generated,
        "reference_text": reference_text,
        "translated_output": translated,
        "rouge_metrics": rouge_metrics,
        "readability_metrics": readability_metrics,
        "elapsed_time": round(elapsed_time, 2),
        "model_name": SUMMARIZATION_MODEL_NAME if task_type == "summarization" else PARAPHRASING_MODEL_NAME,
        "task": task_type,
        "reference_status": reference_status,
        "spider_chart_data": spider_chart_data,
        "timestamp": current_timestamp
    }

    log_history(email, task_type, result["model_name"], request.text, result)
    return result

# --------------------------
# Feedback Endpoint
# --------------------------
@app.post("/feedback")
def submit_feedback(request: FeedbackRequest, auth_payload: dict = Depends(authenticate_user)):
    email = auth_payload["sub"]
    success = update_history_feedback(email, request.timestamp, request.score, request.comment)
    if success:
        return {"message": "Feedback successfully recorded."}
    else:
        raise HTTPException(status_code=404, detail="History entry not found or update failed.")

# --------------------------
# History Endpoint
# --------------------------
@app.get("/history")
def get_history(auth_payload: dict = Depends(authenticate_user)):
    email = auth_payload["sub"]
    history_data = get_user_history(email)
    return {"history": history_data}

# --------------------------
# Admin Endpoints
# --------------------------
# backend.py - Add the following functions

# --- Admin Content Curation Endpoints ---

@app.get("/admin/all_history")
def get_all_history(auth_payload: dict = Depends(authenticate_admin)):
    """Admin endpoint to fetch ALL processing history for all users."""
    logging.info(f"Admin {auth_payload['sub']} fetching all history.")
    
    all_history_records = []

    # ✅ Pull history from MOCK_USERS_DB (not DB)
    for email, user_data in MOCK_USERS_DB.items():
        for record in user_data.get("history", []):
            record_copy = record.copy()
            record_copy["user_email"] = email  # include who owns it
            all_history_records.append(record_copy)

    # Sort by timestamp (newest first)
    all_history_records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return {"history": all_history_records}


class ContentUpdate(BaseModel):
    user_email: str # The email of the user who owns the record
    generated_output: str

@app.put("/admin/content/{timestamp}")
def update_history_entry(timestamp: str, payload: dict, auth_payload: dict = Depends(authenticate_admin)):
    """Admin endpoint to update a specific user's history entry."""
    logging.info(f"Admin {auth_payload['sub']} updating history entry at {timestamp}.")

    target_email = payload.get("user_email")
    if not target_email:
        raise HTTPException(status_code=400, detail="Missing user email in request body.")

    user_history = MOCK_USERS_DB.get(target_email, {}).get("history", [])
    record = next((r for r in user_history if r.get("timestamp") == timestamp), None)

    # ✅ Check if record exists
    if not record:
        raise HTTPException(status_code=404, detail="History record not found.")

    # ✅ Update fields
    for key, value in payload.items():
        if key in record:
            record[key] = value

    logging.info(f"Updated record for {target_email} at {timestamp}.")
    return {"message": "History entry updated successfully."}


import mysql.connector
from fastapi import HTTPException, Depends, Query

# ----------------------------------------------------
# Content Deletion Endpoint 
# ----------------------------------------------------
# NOTE: This function requires the 'get_db_connection' function, 
# 'app' instance, and 'authenticate_user' dependency to be defined in the main application file.

# The existing delete_history_entry function (defined earlier in your file)
def delete_history_entry(email: str, timestamp: str) -> bool:
    """Deletes a history entry by email and timestamp."""
    if email not in MOCK_USERS_DB or 'history' not in MOCK_USERS_DB[email]:
        return False
    
    history = MOCK_USERS_DB[email]['history']
    original_len = len(history)
    
    # Filter out the entry with the matching timestamp
    MOCK_USERS_DB[email]['history'] = [
        entry for entry in history if entry.get('timestamp') != timestamp
    ]
    
    new_len = len(MOCK_USERS_DB[email]['history'])
    
    if new_len < original_len:
        save_user_data()
        logging.info(f"History entry deleted for {email} at {timestamp}.")
        return True
    logging.warning(f"History entry with timestamp {timestamp} not found for user {email} during deletion.")
    return False

# ----------------------------------------------------
# Content Deletion Endpoint (FIXED)
# ----------------------------------------------------

@app.delete("/admin/content/{record_id}") 
async def delete_content(
    record_id: str, 
    user_email: str = Query(..., description="Email of the user whose content is being deleted"),
    auth_payload: dict = Depends(authenticate_user)
):
    """
    Admin endpoint to securely delete a history record using its timestamp ID 
    and the associated user's email, using the file-based mock database.
    """
    # Use authenticate_admin dependency for stricter admin check
    # Or, keep the explicit check if authenticate_admin is not used as a direct dependency
    if auth_payload.get("role") != "admin": # Check the role from the auth payload
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    
    # *** FIX: Replace MySQL interaction with call to mock database function ***
    
    success = delete_history_entry(user_email, record_id)

    if not success:
        # If delete_history_entry returns False, the record was not found or failed to delete.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"No record found matching ID {record_id} and email {user_email}. It may have already been deleted."
        )

    # Note: save_user_data() is called within delete_history_entry on success.

    return {"message": f"Content ID {record_id} for user {user_email} deleted successfully from mock DB."}
@app.get("/admin/analytics")
def admin_analytics(auth_payload: dict = Depends(authenticate_admin)):
    # Compute analytics expected by frontend
    all_data = list_all_history()
    unique_users = len([email for email in MOCK_USERS_DB.keys()])
    total_generated_texts = sum(len(user.get('history', [])) for user in MOCK_USERS_DB.values())
    model_usage_counts: Dict[str, int] = {}
    top_used_model = None
    for email, data in MOCK_USERS_DB.items():
        for entry in data.get('history', []):
            model_name = entry.get('model', 'unknown')
            model_usage_counts[model_name] = model_usage_counts.get(model_name, 0) + 1
    if model_usage_counts:
        top_used_model = max(model_usage_counts.items(), key=lambda x: x[1])[0]
    stats = {
        "unique_users": unique_users,
        "total_generated_texts": total_generated_texts,
        "top_used_model": top_used_model or "N/A",
        "model_usage_counts": model_usage_counts
    }
    return {"stats": stats}

def _user_list_for_admin():
    users = []
    for idx, (email, data) in enumerate(MOCK_USERS_DB.items(), start=1):
        users.append({"id": idx, "email": email, "role": data.get('role', 'user')})
    return users

def _get_email_by_user_id(user_id: int) -> Optional[str]:
    # stable ordering: enumerate over sorted emails for deterministic ids
    emails = list(MOCK_USERS_DB.keys())
    if user_id < 1 or user_id > len(emails):
        return None
    return emails[user_id - 1]

@app.get("/admin/users")
def admin_users(auth_payload: dict = Depends(authenticate_admin)):
    users = _user_list_for_admin()
    return {"users": users}

@app.post("/admin/user/{user_id}/promote")
def admin_promote_user(user_id: int, auth_payload: dict = Depends(authenticate_admin)):
    email = _get_email_by_user_id(user_id)
    if not email:
        raise HTTPException(status_code=404, detail="User not found")
    MOCK_USERS_DB[email]['role'] = 'admin'
    save_user_data()
    return {"message": f"{email} promoted to admin."}

@app.post("/admin/user/{user_id}/demote")
def admin_demote_user(user_id: int, auth_payload: dict = Depends(authenticate_admin)):
    email = _get_email_by_user_id(user_id)
    if not email:
        raise HTTPException(status_code=404, detail="User not found")
    MOCK_USERS_DB[email]['role'] = 'user'
    save_user_data()
    return {"message": f"{email} demoted to user."}

@app.delete("/admin/user/{user_id}")
def admin_delete_user(user_id: int, auth_payload: dict = Depends(authenticate_admin)):
    email = _get_email_by_user_id(user_id)
    if not email:
        raise HTTPException(status_code=404, detail="User not found")
    # prevent deleting the caller admin themselves (optional safety)
    caller_email = auth_payload['sub']
    if email == caller_email:
        raise HTTPException(status_code=400, detail="Cannot delete yourself.")
    MOCK_USERS_DB.pop(email, None)
    save_user_data()
    return {"message": f"User {email} deleted."}

@app.get("/admin/feedback_summary")
def admin_feedback_summary(auth_payload: dict = Depends(authenticate_admin)):
    likes = 0
    dislikes = 0
    total_feedback = 0
    recent_logins = RECENT_LOGINS[:10]
    # aggregate feedback across all users
    for email, data in MOCK_USERS_DB.items():
        for entry in data.get('history', []):
            score = entry.get('feedback_score')
            if score is not None:
                total_feedback += 1
                if score == 1:
                    likes += 1
                elif score == -1:
                    dislikes += 1
    feedback_stats = {
        "likes": likes,
        "dislikes": dislikes,
        "total_feedback": total_feedback,
        "recent_logins": recent_logins
    }
    return {"feedback_stats": feedback_stats}

# --- NEW ENDPOINT: Detailed Feedback ---
@app.get("/admin/detailed_feedback")
def admin_detailed_feedback(auth_payload: dict = Depends(authenticate_admin)):
    """
    Retrieves the top 10 most positive (score=1) and top 10 most negative (score=-1) feedback entries.
    This replaces the 404 error for the "Top 10 Positive and Negative Feedback" section.
    """
    all_data = list_all_history()
    
    # Filter for entries with feedback
    feedback_entries = [
        entry for entry in all_data 
        if entry.get('feedback_score') is not None and entry.get('feedback_comment')
    ]
    
    # Sort and slice
    positive_feedback = [
        entry for entry in feedback_entries 
        if entry['feedback_score'] == 1
    ][:10]
    
    negative_feedback = [
        entry for entry in feedback_entries 
        if entry['feedback_score'] == -1
    ][:10]

    # The frontend expects a successful response, even if the lists are empty
    return {
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback
    }

# --- NEW ENDPOINT: Content Curation ---
@app.get("/admin/curation_content")
def admin_curation_content(auth_payload: dict = Depends(authenticate_admin)):
    """
    Retrieves content records suitable for curation (e.g., based on low readability or a specific model).
    This replaces the 404 error for the "Content Curation" section.
    
    For a mock implementation, we return the last 10 generated outputs.
    A real system might filter based on specific metrics like a low Flesch score.
    """
    all_data = list_all_history()
    
    # Filter for entries that have a generated output and are not just batch requests
    curation_candidates = [
        {
            "id": idx + 1,
            "timestamp": entry.get('timestamp'),
            "user_email": entry.get('user_email'),
            "task": entry.get('task'),
            "model": entry.get('model'),
            "generated_output": entry.get('generated_output', 'N/A'),
            "readability_score": entry.get('readability_metrics', {}).get('flesch_reading_ease', 'N/A')
        }
        for idx, entry in enumerate(all_data) 
        if entry.get('generated_output') and entry.get('task') in ['summarization', 'paraphrasing']
    ]
    
    # Return the 10 most recent content records for curation
    return {"curation_records": curation_candidates[:10]}


# --------------------------
# Batch Processing Endpoint (Simulated)
# --------------------------
@app.post("/batch_process_csv")
def batch_process_csv(
    file: UploadFile = File(...),
    task: str = Query(...),
    language: str = Query(...),
    auth_payload: dict = Depends(authenticate_user)
):
    email = auth_payload["sub"]
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    try:
        content = file.file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'text' column.")
        log_history(
            email,
            f"Batch {task}",
            "N/A (Batch)",
            f"CSV file: {file.filename}, Rows: {len(df)}",
            {"status": "Processing initiated (simulated background task)"}
        )
        return {
            "message": f"Batch {task} initiated successfully for {len(df)} rows. Results will be saved to history upon completion.",
            "rows": len(df),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        logging.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")