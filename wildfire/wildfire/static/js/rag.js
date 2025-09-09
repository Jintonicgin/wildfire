// rag.js — safe single-call, debounce, better errors
(function () {
  // --- prevent duplicate bindings on hot-reload ---
  if (window.__RAG_JS_BOUND__) return;
  window.__RAG_JS_BOUND__ = true;

  const $ = (sel) => document.querySelector(sel);

  // ---------- DOM ----------
  const ragQuery   = $("#ragQuery");
  const topK       = $("#topK");
  const threshold  = $("#threshold");
  const btnSearch  = $("#btnSearch");
  const results    = $("#results");
  const toastEl    = $("#toast");

  // ---------- UI helpers ----------
  function toast(msg, ms = 2200) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.remove("hidden");
    setTimeout(() => toastEl.classList.add("hidden"), ms);
  }
  function setDisabled(el, on) { if (el) el.disabled = !!on; }
  function escapeHtml(s) {
    return (s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ---------- Renderers ----------
  function renderAnswer(answer, sources) {
    const srcHtml =
      (sources || [])
        .map((s) => {
          const score = s.rerank_score != null ? s.rerank_score : s.score;
          return `
            <div class="src-card">
              <div class="src-head">
                <span class="tag">${escapeHtml(s.id || "")}</span>
                <strong class="src-name">${escapeHtml(s.source || "doc")}</strong>
              </div>
              <pre class="src-snippet">${escapeHtml(s.text || "")}</pre>
            </div>`;
        })
        .join("") || `<div class="muted small">출처가 없습니다.</div>`;

    results.innerHTML = `
      <div class="answer-card">
        <div class="answer-title">응답</div>
        <div class="answer-body">${escapeHtml(answer || "")}</div>
      </div>
      <div class="sources-wrap">
        <div class="sources-title">참고 문맥</div>
        ${srcHtml}
      </div>`;
  }

  // ---------- API ----------
  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(body || {}),
    });

    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      const err = new Error(txt || `HTTP ${r.status}`);
      err.status = r.status;
      throw err;
    }
    return r.json();
  }

  // ---------- Actions ----------
  let asking = false;              // in-flight guard
  let enterTimer = null;           // debounce for Enter

  async function ask() {
    if (asking) return;            // 이미 요청 중이면 무시

    const q   = (ragQuery && ragQuery.value.trim()) || "";
    const k   = parseInt((topK && topK.value) || "5", 10);
    const thr = parseFloat((threshold && threshold.value) || "0.25");

    if (!q) {
      toast("질문을 입력하세요");
      return;
    }

    asking = true;
    setDisabled(btnSearch, true);
    results.innerHTML = `<div class="muted small">검색 및 생성 중…</div>`;

    try {
      const data = await postJSON("/api/rag/ask", { query: q, k, thr });
      if (data.ok) {
        renderAnswer(data.answer, data.sources);
      } else if (data.error === "NO_CONTEXT") {
        results.innerHTML = `<div class="muted small">문맥을 찾지 못했습니다. 임계값을 낮추거나 인덱스를 확인하세요.</div>`;
      } else {
        results.innerHTML = `<div class="muted small">요청 실패</div>`;
        toast(data.error || "요청 실패");
      }
    } catch (err) {
      // 502 등 서버 게이트웨이 오류 메시지 일부 표기
      const msg =
        err?.status === 502 ? "백엔드 게이트웨이 오류(502)" :
        err?.message ? (err.message.slice(0, 180) + (err.message.length > 180 ? "…" : "")) :
        "요청 중 오류";
      results.innerHTML = `<div class="muted small">${escapeHtml(msg)}</div>`;
      toast(msg);
      console.error(err);
    } finally {
      setDisabled(btnSearch, false);
      asking = false;
    }
  }

  // ---------- Bindings ----------
  if (btnSearch) {
    // 한 번만 묶이도록 옵션 사용
    btnSearch.addEventListener("click", ask, { passive: true });
  }

  if (ragQuery) {
    // Enter 키 디바운스(200ms) + 중복 방지
    ragQuery.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        clearTimeout(enterTimer);
        enterTimer = setTimeout(() => ask(), 200);
      }
    }, { passive: false });
  }
})();