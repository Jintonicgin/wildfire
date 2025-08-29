(function () {
  const $ = (sel) => document.querySelector(sel);

  // ---------- DOM ----------
  const ragQuery   = $("#ragQuery");
  const topK       = $("#topK");
  const threshold  = $("#threshold");
  const btnSearch  = $("#btnSearch");
  const results    = $("#results");
  const toastEl    = $("#toast");

  // ---------- UI helpers ----------
  function toast(msg, ms = 2000) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.remove("hidden");
    setTimeout(() => toastEl.classList.add("hidden"), ms);
  }

  function setDisabled(el, on) {
    if (!el) return;
    el.disabled = !!on;
  }

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
                <span class="score">관련도: ${score != null ? score.toFixed(3) : "-"}</span>
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
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  // ---------- Actions ----------
  async function ask() {
    const q   = (ragQuery && ragQuery.value.trim()) || "";
    const k   = parseInt((topK && topK.value) || "5", 10);
    const thr = parseFloat((threshold && threshold.value) || "0.25");

    if (!q) {
      toast("질문을 입력하세요");
      return;
    }

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
      results.innerHTML = `<div class="muted small">요청 중 오류</div>`;
      toast("요청 중 오류");
      console.error(err);
    } finally {
      setDisabled(btnSearch, false);
    }
  }

  // ---------- Bindings ----------
  if (btnSearch) btnSearch.addEventListener("click", ask);

  // 엔터로 전송
  if (ragQuery) {
    ragQuery.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ask();
      }
    });
  }
})();
