# -*- coding: utf-8 -*-
from __future__ import annotations

from uuid import uuid4
import os
import json
import requests
from flask import Blueprint, request, jsonify, render_template
from langdetect import detect, LangDetectException

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

bp = Blueprint("gai", __name__, url_prefix="/gai")

# ===================== 설정 =====================
GAI_BACKEND = (os.getenv("GAI_BACKEND_URL", "").strip() or "").rstrip("/")  # 빈 문자열 허용
GAI_TIMEOUT = int(os.getenv("GAI_TIMEOUT", "120"))

# (로컬 폴백) 요약/번역 모델
TEXT_GEN_MODEL = os.getenv("GAI_TEXT_GEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # 미사용(참고용)
SUM_MODEL      = os.getenv("GAI_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
NLLB_MODEL     = os.getenv("GAI_TRANS_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_EN        = os.getenv("NLLB_EN", "eng_Latn")
NLLB_KO        = os.getenv("NLLB_KO", "kor_Hang")

GEN_MAX_NEW   = int(os.getenv("GAI_GEN_MAX_NEW_TOKENS", "2048"))
TEMP_DEFAULT  = float(os.getenv("GAI_TEMPERATURE", "0.7"))
TOP_P         = float(os.getenv("GAI_TOP_P", "0.9"))

SUM_MINLEN    = int(os.getenv("GAI_SUMMARY_MINLEN", "20"))
SUM_MAXLEN    = int(os.getenv("GAI_SUMMARY_MAXLEN", "80"))
SUM_LEN_TRIG  = int(os.getenv("GAI_SUMMARY_LENGTH_THRESHOLD", "280"))

# ====== Ollama (로컬 생성) ======
USE_OLLAMA         = (os.getenv("USE_OLLAMA", "true").lower() == "true")
OLLAMA_URL         = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL       = os.getenv("LLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
OLLAMA_TOP_P       = float(os.getenv("LLAMA_TOP_P", "0.9"))

# ====== RAG 검색 파라미터(클라우드 호출에 사용) ======
RAG_REQUIRE_CONTEXT  = os.getenv("RAG_REQUIRE_CONTEXT", "true").lower() == "true"
RAG_TOP_K            = int(os.getenv("RAG_TOP_K", "5"))
RAG_SCORE_THRESHOLD  = float(os.getenv("RAG_SCORE_THRESHOLD", "0.25"))

# 세션 메모리
_session_histories: dict[str, list[dict[str, str]]] = {}
_MAX_TURNS = 20

# ===================== 로컬 폴백 전역(요약/번역) =====================
from typing import Any, Optional

_summarizer: Optional[Any] = None  # transformers pipeline; 간단히 Any로 둠
_nllb_tok: Optional[PreTrainedTokenizerBase] = None
_nllb_mdl: Optional[PreTrainedModel] = None
_device: Optional[str] = None


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
    t = (text or "").lower()
    to_en = ["영어로", "영문으로", "to english", "영작", "translate to english"]
    to_ko = ["한국어로", "한글로", "to korean", "국문으로", "번역해줘", "번역", "translate to korean"]
    if any(h in t for h in to_en):
        return "ko->en"
    if any(h in t for h in to_ko):
        return "en->ko"
    return None

def want_summary(text: str) -> bool:
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


# ===================== 로컬 폴백 로더/가드 =====================
def ensure_local_loaded() -> None:
    """요약/번역 파이프라인을 필요 시 로드. 호출 후 전역들이 None 아님을 보장."""
    global _summarizer, _nllb_tok, _nllb_mdl, _device

    if _device is None:
        _device = pick_device()

    # transformers pipeline()의 device 인자:
    #  - GPU/MPS는 정수 인덱스(0), CPU는 -1
    dev_index = 0 if _device in ("cuda", "mps") else -1

    if _summarizer is None:
        _summarizer = pipeline(
            task="summarization",
            model=SUM_MODEL,
            device=dev_index,
        )

    if _nllb_tok is None or _nllb_mdl is None:
        _nllb_tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _nllb_mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        if _device != "cpu":
            _nllb_mdl = _nllb_mdl.to(_device)

def _need_nllb() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """토크나이저/모델이 None이 아님을 보장(정적 분석 경고 제거용)."""
    ensure_local_loaded()
    assert _nllb_tok is not None and _nllb_mdl is not None, "NLLB is not loaded"
    return _nllb_tok, _nllb_mdl

def _need_summarizer() -> Any:
    ensure_local_loaded()
    assert _summarizer is not None, "Summarizer is not loaded"
    return _summarizer


# ===================== 로컬 번역/요약 =====================
def local_translate(text: str, direction: str) -> str:
    tok, mdl = _need_nllb()
    # direction: "ko->en" | "en->ko"
    src = NLLB_KO if direction == "ko->en" else NLLB_EN
    tgt = NLLB_EN if direction == "ko->en" else NLLB_KO

    # 소스 언어 설정 (NLLB 전용)
    tok.src_lang = src  # PreTrainedTokenizerFast 기반에서 지원

    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if _device and _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    forced_bos_id = tok.convert_tokens_to_ids(tgt)
    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            forced_bos_token_id=forced_bos_id,
        )
    out = tok.batch_decode(out_ids, skip_special_tokens=True)
    return (out[0] if out else "").strip()

def local_summarize(text: str) -> str:
    summarizer = _need_summarizer()
    out = summarizer(
        text,
        max_length=SUM_MAXLEN,
        min_length=SUM_MINLEN,
        do_sample=False,
        truncation=True,
    )
    # pipeline 출력은 보통 [{"summary_text": "..."}]
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return (out[0].get("summary_text") or "").strip()
    if isinstance(out, dict):
        return (out.get("summary_text") or "").strip()
    return str(out).strip()


# ===================== RAG 컨텍스트 가져오기 =====================
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

def fetch_rag_context(query: str, *, k: int | None = None, thr: float | None = None) -> tuple[str | None, list[dict] | None, dict | None]:
    if not GAI_BACKEND:
        return None, None, {"error": "NO_CLOUD_BACKEND_CONFIGURED"}

    url = f"{GAI_BACKEND}/api/rag/search"
    payload = {
        "query": (query or "").strip(),
        "k": int(k or RAG_TOP_K),
        "thr": float(thr if thr is not None else RAG_SCORE_THRESHOLD),
        "return_context": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=GAI_TIMEOUT)
        if r.status_code == 200:
            j = r.json() or {}
            if not j.get("ok"):
                return None, None, {"error": j}
            ctx = j.get("context") or ""
            src = j.get("sources") or []
            if not ctx.strip():
                return None, None, {"error": "NO_RAG_CONTEXT"}
            return ctx, src, None

        try:
            j = r.json()
        except Exception:
            j = {"error": r.text}
        return None, None, {"status_code": r.status_code, "error": j}
    except requests.exceptions.RequestException as e:
        return None, None, {"error": f"RAG_BACKEND_REQUEST_FAILED: {e}"}

def local_chat_with_rag_ollama(user_text: str, temperature: float | None, sid: str | None) -> tuple[str | None, dict | None]:
    ctx, _, err = fetch_rag_context(user_text, k=RAG_TOP_K, thr=RAG_SCORE_THRESHOLD)
    if err or not ctx:
        if RAG_REQUIRE_CONTEXT:
            return None, {"error": err or "NO_RAG_CONTEXT"}

    sys = (
        "당신은 친절하고 자세하게 답하는 한국어 비서입니다. 항상 한국어로만 대답하세요."
        "답변은 자세하게 설명해주세요."
        "영어 단어를 사용하지 마세요."
        "한자, 영어, 일본어, 러시아어를 사용하지 마세요. "
        "다음에 제공되는 참고 문맥을 최우선으로 따라서 답변하세요. 문맥에 없는 내용은 ‘제공된 정보로는 확인할 수 없습니다.’라고 답하세요. "
        "반드시 한국어 어휘와 문장부호를 사용하세요."
        "문장이 끊기지 않게 답변하세요."
    )

    prompt = (
        f"[시스템]\n{sys}\n\n"
        + (f"[참고 문맥(context)]\n{ctx}\n\n" if ctx else "")
        + f"[사용자]\n{user_text}\n\n"
        f"[도우미]\n"
    )

    reply = _ollama_chat(
        prompt,
        temperature=temperature if temperature is not None else TEMP_DEFAULT,
        top_p=TOP_P,
        timeout=GAI_TIMEOUT,
    )

    _push_hist(sid, "user", user_text)
    _push_hist(sid, "assistant", reply)
    return reply.strip(), None


# ===================== Flask routes =====================
@bp.get("")
def gai():
    return render_template("nav_page/gai.html")


@bp.post("/chat")
def api_gai_chat():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    temperature = data.get("temperature", None)

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

    mode = auto_route(text)

    try:
        if mode.startswith("translate:"):
            direction = mode.split(":", 1)[1]  # "ko->en" or "en->ko"
            if GAI_BACKEND:
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
                out = local_translate(text, direction)
                resp = jsonify({"ok": True, "reply": out, "mode": "translate (local-fallback)", "session_id": sid})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 200

        elif mode == "summarize":
            if GAI_BACKEND:
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
                out = local_summarize(text)
                resp = jsonify({"ok": True, "reply": out, "mode": "summarize (local-fallback)", "session_id": sid})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 200

        else:
            if not USE_OLLAMA:
                resp = jsonify({"ok": False, "error": "OLLAMA_DISABLED"})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, 503

            reply, err = local_chat_with_rag_ollama(text, temperature, sid)
            if err or not reply:
                status = 422 if (err and "NO_RAG_CONTEXT" in str(err)) else 502
                resp = jsonify({"ok": False, "error": err or "RAG_OR_GEN_FAILED"})
                if new_sid:
                    resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
                return resp, status

            resp = jsonify({"ok": True, "reply": reply, "mode": "chat (ollama-with-rag)", "session_id": sid})
            if new_sid:
                resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
            return resp, 200

    except requests.exceptions.RequestException as e:
        if mode == "chat":
            resp = jsonify({"ok": False, "error": f"RAG_BACKEND_REQUEST_FAILED: {e}"})
            if new_sid:
                resp.set_cookie("gai_session_id", sid, max_age=60*60*24*7, samesite="Lax")
            return resp, 502

        # 번역/요약만 로컬 폴백 허용
        try:
            if mode.startswith("translate:"):
                direction = mode.split(":", 1)[1]
                out = local_translate(text, direction)
                j = {"ok": True, "reply": out, "mode": "translate (local-fallback)", "error_proxy": str(e), "session_id": sid}
            elif mode == "summarize":
                out = local_summarize(text)
                j = {"ok": True, "reply": out, "mode": "summarize (local-fallback)", "error_proxy": str(e), "session_id": sid}
            else:
                j = {"ok": False, "error": f"CHAT_REQUIRES_RAG_CONTEXT: {e}", "session_id": sid}
                resp = jsonify(j)
                if new_sid: resp.set_cookie("gai_session_id", sid, max_age=604800, samesite="Lax")
                return resp, 502
            resp = jsonify(j)
            if new_sid: resp.set_cookie("gai_session_id", sid, max_age=604800, samesite="Lax")
            return resp, 200
        except Exception as e_local:
            resp = jsonify({"ok": False, "error": f"BACKEND_REQUEST_FAILED: {e}; LOCAL_FALLBACK_FAILED: {e_local}"})
            if new_sid: resp.set_cookie("gai_session_id", sid, max_age=604800, samesite="Lax")
            return resp, 502


@bp.get("/healthz")
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