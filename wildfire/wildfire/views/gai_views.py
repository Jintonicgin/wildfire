from uuid import uuid4
import os
import requests
from flask import Blueprint, request, jsonify, render_template
from langdetect import detect, LangDetectException

# 로컬 폴백용
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

bp = Blueprint("gai", __name__)

# ====== 설정 ======
GAI_BACKEND = (os.getenv("GAI_BACKEND_URL", "") or "").rstrip("/")
GAI_TIMEOUT = int(os.getenv("GAI_TIMEOUT", "30"))

TEXT_GEN_MODEL = os.getenv("GAI_TEXT_GEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SUM_MODEL      = os.getenv("GAI_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
NLLB_MODEL     = os.getenv("GAI_TRANS_MODEL", "facebook/nllb-200-distilled-600M")

NLLB_EN = os.getenv("NLLB_EN", "eng_Latn")
NLLB_KO = os.getenv("NLLB_KO", "kor_Hang")

GEN_MAX_NEW   = int(os.getenv("GAI_GEN_MAX_NEW_TOKENS", "64"))
TEMP_DEFAULT  = float(os.getenv("GAI_TEMPERATURE", "0.7"))
TOP_P         = float(os.getenv("GAI_TOP_P", "0.9"))

SUM_MINLEN    = int(os.getenv("GAI_SUMMARY_MINLEN", "20"))
SUM_MAXLEN    = int(os.getenv("GAI_SUMMARY_MAXLEN", "80"))
SUM_LEN_TRIG  = int(os.getenv("GAI_SUMMARY_LENGTH_THRESHOLD", "280"))

_session_histories = {}
_MAX_TURNS = 20

# ====== 로컬 폴백 전역 ======
_text_gen = None
_summarizer = None
_nllb_tok = None
_nllb_mdl = None
_device = None

# ====== 유틸 ======
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text or "")
    except LangDetectException:
        return "unknown"

def want_translation(text: str):
    """명시적 번역 힌트만 ko->en / en->ko 감지. 그 외는 None."""
    t = (text or "").lower()
    to_en = ["영어로", "영문으로", "to english", "영작", "translate to english"]
    to_ko = ["한국어로", "한글로", "to korean", "국문으로", "번역해줘", "번역", "translate to korean"]
    if any(h in t for h in to_en):
        return "ko->en"
    if any(h in t for h in to_ko):
        return "en->ko"
    return None

def want_summary(text: str) -> bool:
    """요약 힌트가 있거나, (번역 의도 없고) 길이가 길면 요약."""
    t = (text or "").lower()
    summary_hints = ["요약", "summary", "summarize", "요약해", "한줄요약", "짧게"]
    if any(h in t for h in summary_hints):
        return True
    if want_translation(text):
        return False
    return len(t) > SUM_LEN_TRIG

def auto_route(text: str) -> str:
    tr = want_translation(text)
    if tr:
        return f"translate:{tr}"
    if want_summary(text):
        return "summarize"
    return "chat"

# ====== 로컬 폴백 로더/실행 ======
def ensure_local_loaded():
    global _text_gen, _summarizer, _nllb_tok, _nllb_mdl, _device
    if _device is None:
        _device = pick_device()

    dev_index = 0 if _device in ("cuda", "mps") else -1

    if _text_gen is None:
        _text_gen = pipeline(
            "text-generation",
            model=TEXT_GEN_MODEL,
            device=dev_index,
            torch_dtype=(torch.float16 if _device in ("cuda", "mps") else None),
        )

    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model=SUM_MODEL,
            device=dev_index,
        )

    if _nllb_tok is None or _nllb_mdl is None:
        _nllb_tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _nllb_mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        if _device != "cpu":
            _nllb_mdl = _nllb_mdl.to(_device)

def local_translate(text: str, direction: str) -> str:
    ensure_local_loaded()
    src = NLLB_EN if direction == "en->ko" else NLLB_KO
    tgt = NLLB_KO if direction == "en->ko" else NLLB_EN
    _nllb_tok.src_lang = src
    inputs = _nllb_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}
    forced_bos_id = _nllb_tok.convert_tokens_to_ids(tgt)
    with torch.no_grad():
        out_ids = _nllb_mdl.generate(
            **inputs, max_new_tokens=256, num_beams=4, forced_bos_token_id=forced_bos_id
        )
    return _nllb_tok.batch_decode(out_ids, skip_special_tokens=True)[0]

def local_summarize(text: str) -> str:
    ensure_local_loaded()
    out = _summarizer(
        text, max_length=SUM_MAXLEN, min_length=SUM_MINLEN, do_sample=False, truncation=True
    )
    return out[0]["summary_text"]

def local_chat(text: str, temperature: float | None) -> str:
    ensure_local_loaded()
    temp = float(temperature) if temperature is not None else TEMP_DEFAULT
    prompt = text if text.endswith(":") else text + "\n"
    out = _text_gen(
        prompt,
        max_new_tokens=GEN_MAX_NEW,
        do_sample=True,
        temperature=max(0.2, min(temp, 1.0)),
        top_p=TOP_P,
    )
    full = out[0]["generated_text"]
    return full[len(prompt):].strip() if full.startswith(prompt) else full

def _get_hist(sid):
    return _session_histories.get(sid, [])

def _push_hist(sid, role, content):
    if not sid:
      return
    lst = _session_histories.setdefault(sid, [])
    lst.append({"role": role, "content": content})
    if len(lst) > _MAX_TURNS:
        del lst[0:len(lst)-_MAX_TURNS]

def local_chat_with_memory(text: str, temperature: float | None, sid: str | None) -> str:
    """TinyLlama 파이프라인용: 간단한 히스토리 연결"""
    ensure_local_loaded()
    temp = float(temperature) if temperature is not None else TEMP_DEFAULT

    # 간단 템플릿(너무 길어지면 모델이 흔들려서 간결하게 구성)
    hist = _get_hist(sid)
    pieces = []
    for m in hist[-10:]:  # 최근만
        tag = "user" if m["role"] == "user" else "assistant"
        pieces.push = None
        pieces.append(f"{tag}: {m['content']}")
    pieces.append(f"user: {text}")
    prompt = "\n".join(pieces) + "\nassistant:"

    out = _text_gen(
        prompt,
        max_new_tokens=GEN_MAX_NEW,
        do_sample=True,
        temperature=max(0.2, min(temp, 1.0)),
        top_p=TOP_P,
    )
    full = out[0]["generated_text"]
    reply = full[len(prompt):].strip() if full.startswith(prompt) else full

    _push_hist(sid, "user", text)
    _push_hist(sid, "assistant", reply)
    return reply

# ====== Flask routes ======
@bp.get("/gai")
def gai():
    return render_template("nav_page/gai.html")

@bp.post("/api/gai/chat")
def api_gai_chat():
    """
    1) GAI_BACKEND_URL 설정 시: FastAPI 프록시 (★ session_id 전달)
    2) 실패/미설정 시: 로컬 폴백 (★ 메모리 사용)
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    temperature = data.get("temperature", None)

    # ★ 세션 결정: body > cookie > 신규 발급
    sid = data.get("session_id") or request.cookies.get("gai_session_id")
    new_sid = False
    if not sid:
        sid = str(uuid4())
        new_sid = True

    if not text:
        resp = jsonify({"ok": False, "error": "EMPTY_TEXT"})
        if new_sid:
            resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
        return resp, 400

    # --- 1) 프록시 모드 ---
    if GAI_BACKEND:
        try:
            r = requests.post(
                f"{GAI_BACKEND}/api/gai/chat",
                json={"text": text, "temperature": temperature, "session_id": sid},  # ★ 전달
                timeout=GAI_TIMEOUT,
            )
            r.raise_for_status()
            j = r.json()
            resp = jsonify(j)
            # 백엔드가 session_id를 되돌리면 동기화
            if j.get("session_id"):
                sid = j["session_id"]
            if new_sid:
                resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
            return resp, (200 if j.get("ok") else 500)

        except requests.exceptions.RequestException as e:
            fallback_error = str(e)
            # --- 2) 로컬 폴백 ---
            try:
                mode = auto_route(text)
                if mode.startswith("translate:"):
                    direction = mode.split(":", 1)[1]
                    reply = local_translate(text, direction)
                elif mode == "summarize":
                    reply = local_summarize(text)
                else:
                    reply = local_chat_with_memory(text, temperature, sid)

                j = {
                    "ok": True,
                    "reply": reply,
                    "mode": f"{mode} (local-fallback)",
                    "error_proxy": f"BACKEND_REQUEST_FAILED: {fallback_error}",
                    "session_id": sid,
                }
                resp = jsonify(j)
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 200

            except Exception as e_local:
                resp = jsonify({
                    "ok": False,
                    "error": f"BACKEND_REQUEST_FAILED: {fallback_error}; LOCAL_FALLBACK_FAILED: {e_local}"
                })
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 502

    # --- 3) 로컬 전용 모드 ---
    try:
        mode = auto_route(text)
        if mode.startswith("translate:"):
            direction = mode.split(":", 1)[1]
            reply = local_translate(text, direction)
        elif mode == "summarize":
            reply = local_summarize(text)
        else:
            reply = local_chat_with_memory(text, temperature, sid)

        j = {"ok": True, "reply": reply, "mode": mode, "session_id": sid}
        resp = jsonify(j)
        if new_sid:
            resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
        return resp, 200

    except Exception as e:
        resp = jsonify({"ok": False, "error": f"LOCAL_ERROR: {e}"})
        if new_sid:
            resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
        return resp, 500

@bp.get("/api/gai/healthz")
def api_gai_health():
    if GAI_BACKEND:
        try:
            r = requests.get(f"{GAI_BACKEND}/healthz", timeout=5)
            return (r.content, r.status_code, {
                "Content-Type": r.headers.get("Content-Type", "application/json")
            })
        except Exception as e:
            return jsonify({"ok": False, "error": f"HEALTHCHECK_FAIL: {e}"}), 502
    else:
        ensure_local_loaded()
        return jsonify({"ok": True, "device": _device})