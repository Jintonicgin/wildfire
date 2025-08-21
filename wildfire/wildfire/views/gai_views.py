# wildfire/views/gai_views.py
import os
from flask import Blueprint, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException

bp = Blueprint("gai", __name__)

# Globals for pipelines, will be loaded once at startup
_text_gen = None
_summarizer = None
_translator = None
_tok = None

@bp.get("/gai")
def gai():
    return render_template("nav_page/gai.html")

def load_pipelines():
    """
    Loads all generative models into memory.
    This should be called once at application startup.
    """
    global _text_gen, _summarizer, _translator, _tok

    print("Loading all AI/ML models for cloud environment...")

    # Text Generation Model
    if _text_gen is None:
        print("Loading text generation model (TinyLlama)...")
        _text_gen = pipeline(
            "text-generation",
            model=os.getenv("GAI_TEXT_GEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            device_map="auto"
        )
        print("Text generation model loaded.")

    # Summarization Model (Original high-quality version)
    if _summarizer is None:
        print("Loading summarization model (distilbart-cnn-12-6)...")
        _summarizer = pipeline(
            "summarization",
            model=os.getenv("GAI_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6"),
            device_map="auto"
        )
        print("Summarization model loaded.")

    # Translation Model (Original high-quality NLLB version)
    if _translator is None or _tok is None:
        print("Loading translation model (NLLB-600M)...")
        nllb_id = os.getenv("GAI_TRANS_MODEL", "facebook/nllb-200-distilled-600M")
        _tok = AutoTokenizer.from_pretrained(nllb_id, use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
        _translator = AutoModelForSeq2SeqLM.from_pretrained(nllb_id, use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
        print("Translation model loaded.")
    
    print("All AI/ML models loaded.")

def _detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def _want_summary(text: str) -> bool:
    t = text.lower()
    hints = ["요약", "summary", "summarize", "요약해", "한줄요약", "짧게"]
    longish = len(text) > int(os.getenv("GAI_SUMMARY_LENGTH_THRESHOLD", "280"))
    return any(h in t for h in hints) or longish

def _auto_route(text: str) -> str:
    if _want_summary(text):
        return "summarize"
    t = text.lower()
    if any(h in t for h in ["영어로", "to english", "영작", "영문으로"]):
        return "translate:ko->en"
    if any(h in t for h in ["한국어로", "한글로", "to korean", "번역", "번역해줘"]):
        return "translate:en->ko"
    lang = _detect_lang(text)
    if lang == "en":
        return "translate:en->ko"
    return "chat"

def _translate_nllb(text: str, direction: str) -> str:
    # direction: "en->ko" or "ko->en"
    src = os.getenv("NLLB_EN", "eng_Latn") if direction == "en->ko" else os.getenv("NLLB_KO", "kor_Hang")
    tgt = os.getenv("NLLB_KO", "kor_Hang") if direction == "en->ko" else os.getenv("NLLB_EN", "eng_Latn")

    inputs = _tok(text, return_tensors="pt")
    _tok.src_lang = src
    forced_bos_id = _tok.lang_code_to_id[tgt]

    out = _translator.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        forced_bos_token_id=forced_bos_id
    )
    return _tok.batch_decode(out, skip_special_tokens=True)[0]

@bp.post("/api/gai/chat")
def api_gai_chat():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    temperature = float(data.get("temperature", os.getenv("GAI_TEMPERATURE", "0.7")))
    if not text:
        return jsonify({"ok": False, "error": "EMPTY_TEXT"}), 400

    mode = _auto_route(text)
    lang = _detect_lang(text)

    try:
        if mode == "summarize":
            max_len = int(os.getenv("GAI_SUMMARY_MAXLEN", "80"))
            min_len = int(os.getenv("GAI_SUMMARY_MINLEN", "20"))
            out = _summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            reply = out[0]["summary_text"]

        elif mode.startswith("translate:"):
            direction = mode.split(":", 1)[1]
            reply = _translate_nllb(text, direction)

        else:
            max_new = int(os.getenv("GAI_GEN_MAX_NEW_TOKENS", "64"))
            prompt = text if text.endswith(":") else text + "\n"
            out = _text_gen(
                prompt,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=float(os.getenv("GAI_TOP_P", "0.9"))
            )
            full = out[0]["generated_text"]
            reply = full[len(prompt):].strip()

        return jsonify({"ok": True, "reply": reply, "mode": mode, "detected_lang": lang})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500