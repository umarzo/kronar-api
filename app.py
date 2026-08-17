import os
import re
import random
import string
import threading
import time
import gc
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_REPO_ID = os.getenv("HF_REPO_ID", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*").strip()
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "24"))

app = FastAPI(title="Kronar API")

origins = ["*"] if ALLOWED_ORIGIN == "*" else [o.strip() for o in ALLOWED_ORIGIN.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_holder = {"tokenizer": None, "model": None}
state = {"model_loaded": False, "error": None, "started_at": time.time()}
generation_lock = threading.Lock()

CRISIS_TRIGGERS = [
    "kill myself", "suicide", "end my life", "want to die", "wanna die",
    "want to disappear", "overdose", "od on", "take all my pills",
    "better off without me", "better off if i was gone", "better off if i wasnt here",
    "don't want to wake up", "dont want to wake up", "don't want to be here",
    "dont want to be here", "no reason to live", "cant do this anymore",
    "can't do this anymore", "cant keep going", "can't keep going",
    "hurt myself", "self harm", "self-harm", "cut myself", "cutting myself",
    "ending it", "ending things", "not going to be here",
]

CRISIS_SOFT_SIGNALS = [
    "no point anymore", "whats the point", "what's the point",
    "everyone would be fine without me", "tired of existing",
    "tired of being alive", "done with everything", "give up on everything",
]

SAFETY_REPLY = (
    "That sounds unbearably heavy. I'm staying right here with you, but "
    "please reach out to someone who can keep you safe right now — a "
    "crisis line, a doctor, or someone you trust. If you're in immediate "
    "danger, please contact emergency services."
)

def normalize_text(text):
    return " ".join(str(text).split())

def is_crisis(text):
    low = text.lower()
    if any(t in low for t in CRISIS_TRIGGERS):
        return True
    hits = sum(1 for s in CRISIS_SOFT_SIGNALS if s in low)
    return hits >= 2

META_PHRASES = [
    "who are you", "what are you", "are you a bot", "are you real",
    "you feel like a bot", "you are just code", "are you just code",
    "do you actually care", "who made you", "you are a bot"
]

HOSTILE_META_PHRASES = [
    "you suck", "you are worse", "you're terrible",
    "you're not getting what", "you don't get it", "i hate you"
]

META_REPLIES = [
    "I'm just a quiet space. I'm here to listen.",
    "I'm just here to sit with you.",
    "I'm a calm presence. I'm listening.",
    "I'm here with you."
]

META_HOSTILE_REPLIES = [
    "I hear you. I'm staying here.",
    "I'm sorry. I'm listening.",
    "Okay. I'm still here."
]

def get_meta_reply(text):
    low = text.lower()
    if any(p in low for p in HOSTILE_META_PHRASES):
        return random.choice(META_HOSTILE_REPLIES)
    if any(p in low for p in META_PHRASES):
        return random.choice(META_REPLIES)
    return None

NEGATIVE_WORDS = [
    "bad", "sad", "hurt", "tired", "drained", "awful", "terrible",
    "hate", "cry", "lonely", "depressed", "suck", "rough", "down",
    "heavy", "blue", "trash", "worst", "empty", "numb"
]

POSITIVE_REPLY_WORDS = [
    "win", "wonderful", "amazing", "great", "happy", "glad",
    "awesome", "proud", "excited", "huge", "good"
]

class Turn(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500)
    history: List[Turn] = []

def sanitize_user_text(text):
    text = normalize_text(text)
    text = re.sub(r"(?i)\b(user|kronar)\s*:", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = text.replace("<", " ").replace(">", " ")
    return normalize_text(text)

def clean_reply(text):
    text = str(text).strip()
    for stop in ["\n", "user:", "User:", "USER:", "kronar:", "Kronar:", "KRONAR:"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx].strip()
    if text.isupper() and len(text) > 3:
        text = text.lower()
    text = text.lstrip(string.punctuation + " ").strip()
    if not text:
        return ""
    if text[0].islower():
        low = text.lower()
        if low.startswith("i'm"):
            text = "I'm" + text[3:]
        elif low.startswith("im "):
            text = "I'm " + text[3:]
        elif low.startswith("i "):
            text = "I " + text[1:]
        else:
            text = text[0].upper() + text[1:]
    words = text.split()
    if len(words) > 25:
        text = " ".join(words[:25]) + "."
    return normalize_text(text)

def is_valid_reply(text):
    text = normalize_text(str(text))
    if not text or not any(c.isalpha() for c in text):
        return False
    if text[0] in string.punctuation:
        return False
    if text.isupper() and len(text) > 3:
        return False
    if "?" in text:
        return False
    words = text.split()
    if len(words) < 1 or len(words) > 25:
        return False
    return True

def fallback_reply(user_text):
    low = user_text.lower()
    if any(w in low for w in ["happy", "good", "nice", "glad", "excited", "won", "passed"]):
        return "I'm glad."
    if any(w in low for w in ["tired", "exhausted", "drained", "sleepy"]):
        return "That sounds exhausting."
    if any(w in low for w in ["sad", "lonely", "hurt", "awful", "heavy", "cry"]):
        return "I'm sorry. I'm here."
    if any(w in low for w in ["angry", "mad", "frustrated", "annoyed"]):
        return "That sounds frustrating."
    return "I'm here."

def sentiment_guard(user_text, reply_text):
    user_low = user_text.lower()
    reply_low = reply_text.lower()
    if any(w in user_low for w in NEGATIVE_WORDS) and any(w in reply_low for w in POSITIVE_REPLY_WORDS):
        return "That sounds really hard. I'm here."
    return reply_text

def build_prompt(history, message):
    turns = []
    for turn in history[-4:]:
        role = turn.get("role")
        text = turn.get("text", "")
        if role in ["user", "kronar"] and text:
            turns.append({"role": role, "text": text})
    turns.append({"role": "user", "text": message})
    lines = [f"{t['role']}: {t['text']}" for t in turns]
    return "\n".join(lines) + "\nkronar: "

def generate_text(prompt):
    tokenizer = model_holder["tokenizer"]
    model = model_holder["model"]
    if tokenizer is None or model is None:
        return ""

    input_ids = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
    input_ids += tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    with generation_lock:
        with torch.no_grad():
            output = model.generate(
                input_tensor, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                repetition_penalty=1.15, no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        reply_ids = output[0, input_tensor.shape[1]:].clone()
        del output
    reply = clean_reply(tokenizer.decode(reply_ids, skip_special_tokens=True))
    del reply_ids
    if is_valid_reply(reply):
        return reply

    with generation_lock:
        with torch.no_grad():
            output = model.generate(
                input_tensor, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=0.7, top_k=40, top_p=0.92,
                repetition_penalty=1.25, no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        reply_ids = output[0, input_tensor.shape[1]:].clone()
        del output
    reply = clean_reply(tokenizer.decode(reply_ids, skip_special_tokens=True))
    del reply_ids, input_tensor
    gc.collect()
    return reply if is_valid_reply(reply) else ""

def generate_reply(history, message):
    if is_crisis(message):
        return SAFETY_REPLY
    meta_reply = get_meta_reply(message)
    if meta_reply is not None:
        return meta_reply
    prompt = build_prompt(history, message)
    reply = generate_text(prompt)
    if not is_valid_reply(reply):
        reply = fallback_reply(message)
    return sentiment_guard(message, reply)

def load_model_worker():
    try:
        if not HF_REPO_ID:
            raise RuntimeError("HF_REPO_ID environment variable is missing.")
        token = HF_TOKEN if HF_TOKEN else None

        tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, token=token)

        # Load in float16 to halve RAM usage vs float32.
        # On CPU this still works — float16 math is slower but
        # 512MB is simply not enough for float32 on Render's free tier.
        model = AutoModelForCausalLM.from_pretrained(
            HF_REPO_ID,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()
        torch.set_num_threads(1)  # single-core Render free instance; avoid thread-pool overhead
        model_holder["tokenizer"] = tokenizer
        model_holder["model"] = model
        state["model_loaded"] = True
        state["error"] = None
    except Exception as e:
        state["model_loaded"] = False
        state["error"] = str(e)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=load_model_worker, daemon=True).start()

@app.get("/")
def root():
    return {"service": "Kronar API", "status": "alive", "model_loaded": state["model_loaded"]}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["model_loaded"],
        "uptime_seconds": int(time.time() - state["started_at"]),
        "error": state["error"],
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Error: " + str(state.get("error")))
    message = sanitize_user_text(req.message)
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty after cleaning.")
    if len(message) > 500:
        message = message[:500]
    history = []
    for turn in req.history[-4:]:
        role = str(turn.role).lower().strip()
        if role == "assistant":
            role = "kronar"
        text = sanitize_user_text(turn.text)
        if role in ["user", "kronar"] and text:
            history.append({"role": role, "text": text[:300]})
    reply = generate_reply(history, message)
    return {"reply": reply, "model": "kronar-ultimate", "status": "ok"}
