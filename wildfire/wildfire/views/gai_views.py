# -*- coding: utf-8 -*-
from __future__ import annotations

from uuid import uuid4
import os
import requests
from flask import Blueprint, request, jsonify, render_template
from langdetect import detect, LangDetectException

# 로컬 폴백용(HF 파이프라인: 요약/번역에만 사용)
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

bp = Blueprint("gai", __name__)

# ===================== 설정 =====================
GAI_BACKEND = (os.getenv("GAI_BACKEND_URL", "https://lijpcw8himz39f-8080.proxy.runpod.net") or "").rstrip("/")
GAI_TIMEOUT = int(os.getenv("GAI_TIMEOUT", "120"))

# 요약/번역(HF)
TEXT_GEN_MODEL = os.getenv("GAI_TEXT_GEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # (미사용: 로컬 생성은 Ollama)
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

# ====== Ollama (로컬 생성) ======
USE_OLLAMA = (os.getenv("USE_OLLAMA", "true").lower() == "true")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
OLLAMA_TOP_P = float(os.getenv("LLAMA_TOP_P", "0.9"))

# 세션 메모리
_session_histories: dict[str, list[dict[str, str]]] = {}
_MAX_TURNS = 20

# ===================== 로컬 폴백 전역(HF: 요약/번역) =====================
_summarizer = None
_nllb_tok = None
_nllb_mdl = None
_device: str | None = None


# ===================== 유틸 =====================
def pick_device() -> str:
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

def want_translation(text: str) -> str | None:
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


# ===================== Ollama 호출 =====================
def _ollama_chat(prompt: str, *, temperature: float | None = None, top_p: float | None = None,
                 model: str | None = None, timeout: int = 120) -> str:
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": max(0.2, min((temperature if temperature is not None else OLLAMA_TEMPERATURE), 1.0)),
        "top_p": top_p if top_p is not None else OLLAMA_TOP_P,
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json() or {}
    return (j.get("response") or "").strip()


# ===================== 로컬 폴백 로더/실행(HF: 요약/번역만) =====================
def ensure_local_loaded():
    global _summarizer, _nllb_tok, _nllb_mdl, _device
    if _device is None:
        _device = pick_device()

    dev_index = 0 if _device in ("cuda", "mps") else -1

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

def _get_hist(sid: str | None) -> list[dict[str, str]]:
    if not sid:
        return []
    return _session_histories.get(sid, [])

def _push_hist(sid: str | None, role: str, content: str) -> None:
    if not sid:
        return
    lst = _session_histories.setdefault(sid, [])
    lst.append({"role": role, "content": content})
    if len(lst) > _MAX_TURNS:
        del lst[0:len(lst)-_MAX_TURNS]

def local_chat_with_memory(text: str, temperature: float | None, sid: str | None) -> str:
    """
    로컬 대화는 Ollama로 수행. 최근 히스토리를 포함한 간단 프롬프트.
    """
    hist = _get_hist(sid)

    sys = (
        "당신은 한국어로 친절하고 간결하게 답하는 비서입니다. "
        "사실에 기반해 답하고, 모르면 모른다고 말하세요."
    )

    pieces: list[str] = [f"[시스템]\n{sys}"]
    for m in hist[-10:]:
        tag = "사용자" if m["role"] == "user" else "도우미"
        pieces.append(f"[{tag}]\n{m['content']}")
    pieces.append(f"[사용자]\n{text}\n[도우미]\n")

    prompt = "\n".join(pieces)
    reply = _ollama_chat(prompt, temperature=temperature if temperature is not None else TEMP_DEFAULT, top_p=TOP_P)

    _push_hist(sid, "user", text)
    _push_hist(sid, "assistant", reply)
    return reply


# ===================== Flask routes =====================
@bp.get("/gai")
def gai():
    return render_template("nav_page/gai.html")

@bp.post("/api/gai/chat")
def api_gai_chat():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    temperature = data.get("temperature", None)

    # ★ 세션 유지 로직 동일
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

    # ★ 의도 판단 (요약/번역/대화)
    mode = auto_route(text)

    try:
        if mode.startswith("translate:"):
            direction = mode.split(":", 1)[1]  # "ko->en" or "en->ko"
            if GAI_BACKEND:
                # 클라우드 번역 엔드포인트 호출
                r = requests.post(
                    f"{GAI_BACKEND}/api/gai/translate",
                    json={"text": text, "direction": direction},
                    timeout=GAI_TIMEOUT,
                )
                r.raise_for_status()
                j = r.json()
                out = j.get("translation") or ""
                resp = jsonify({"ok": True, "reply": out, "mode": "translate (cloud)", "session_id": sid})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 200
            else:
                # 백엔드 없으면 로컬 NLLB 폴백
                out = local_translate(text, direction)
                resp = jsonify({"ok": True, "reply": out, "mode": "translate (local-fallback)", "session_id": sid})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 200

        elif mode == "summarize":
            if GAI_BACKEND:
                # 클라우드 요약 엔드포인트 호출
                r = requests.post(
                    f"{GAI_BACKEND}/api/gai/summarize",
                    json={"text": text},
                    timeout=GAI_TIMEOUT,
                )
                r.raise_for_status()
                j = r.json()
                out = j.get("summary") or ""
                resp = jsonify({"ok": True, "reply": out, "mode": "summarize (cloud)", "session_id": sid})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 200
            else:
                # 백엔드 없으면 로컬 요약 폴백
                out = local_summarize(text)
                resp = jsonify({"ok": True, "reply": out, "mode": "summarize (local-fallback)", "session_id": sid})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 200

        else:
            # ★ 대화는 항상 로컬 Ollama (의도대로)
            reply = local_chat_with_memory(text, temperature, sid)
            resp = jsonify({"ok": True, "reply": reply, "mode": "chat (local-ollama)", "session_id": sid})
            if new_sid:
                resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
            return resp, 200

    except requests.exceptions.RequestException as e:
        # 클라우드 호출 실패 → 로컬 폴백
        try:
            if mode.startswith("translate:"):
                direction = mode.split(":", 1)[1]
                out = local_translate(text, direction)
                j = {"ok": True, "reply": out, "mode": "translate (local-fallback)", "error_proxy": str(e), "session_id": sid}
            elif mode == "summarize":
                out = local_summarize(text)
                j = {"ok": True, "reply": out, "mode": "summarize (local-fallback)", "error_proxy": str(e), "session_id": sid}
            else:
                out = local_chat_with_memory(text, temperature, sid)
                j = {"ok": True, "reply": out, "mode": "chat (local-fallback)", "error_proxy": str(e), "session_id": sid}
            resp = jsonify(j)
            if new_sid: resp.set_cookie("gai_session_id", sid, max_age=604800, samesite="Lax")
            return resp, 200
        except Exception as e_local:
            resp = jsonify({"ok": False, "error": f"BACKEND_REQUEST_FAILED: {e}; LOCAL_FALLBACK_FAILED: {e_local}"})
            if new_sid: resp.set_cookie("gai_session_id", sid, max_age=604800, samesite="Lax")
            return resp, 502

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
        ensure_local_loaded()  # 요약/번역 파이프라인 준비 상태 확인용
        return jsonify({
            "ok": True,
            "device": _device,
            "use_ollama": USE_OLLAMA,
            "ollama_model": OLLAMA_MODEL
        })