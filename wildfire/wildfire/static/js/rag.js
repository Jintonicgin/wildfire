(() => {
  const chatWindow = document.getElementById("chatWindow");
  const userInput  = document.getElementById("userInput");
  const sendBtn    = document.getElementById("sendBtn");
  const btnIndex   = document.getElementById("btnIndex");
  const btnClear   = document.getElementById("btnClear");
  const indexStatus = document.getElementById("indexStatus");

  const collectionEl = document.getElementById("collectionId");
  const filesEl      = document.getElementById("files");
  const chunkSizeEl  = document.getElementById("chunkSize");
  const overlapEl    = document.getElementById("chunkOverlap");

  // 세션 고정 (프론트에서 유지)
  let sessionId = localStorage.getItem("rag_session_id");
  if (!sessionId) {
    sessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
    localStorage.setItem("rag_session_id", sessionId);
  }

  function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function bubble(role, text) {
    const row = document.createElement("div");
    row.className = `chat-message ${role}`;
    if (role === "bot") {
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      const img = document.createElement("img");
      img.src = "/static/img/bot01.png";
      avatar.appendChild(img);
      row.appendChild(avatar);
    }
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    row.appendChild(b);
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  function loaderBubble() {
    const row = document.createElement("div");
    row.className = "chat-message bot bot-loading";
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    const img = document.createElement("img");
    img.src = "/static/img/bot01.png";
    avatar.appendChild(img);

    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = "검색 및 생성 중…";

    row.appendChild(avatar);
    row.appendChild(b);
    chatWindow.appendChild(row);
    scrollToBottom();
    return {
      remove() { row.remove(); }
    };
  }

  // 인덱싱
  btnIndex?.addEventListener("click", async () => {
    const collection_id = (collectionEl.value || "default").trim();
    const files = filesEl.files;
    const chunk_size = Number(chunkSizeEl.value || 800);
    const chunk_overlap = Number(overlapEl.value || 120);
    if (!files || !files.length) {
      indexStatus.textContent = "파일을 선택하세요.";
      return;
    }
    const fd = new FormData();
    fd.append("collection_id", collection_id);
    fd.append("chunk_size", chunk_size);
    fd.append("chunk_overlap", chunk_overlap);
    for (const f of files) fd.append("files", f);

    indexStatus.textContent = "인덱싱 중…";
    try {
      const res = await fetch("/api/rag/index", { method: "POST", body: fd });
      const j = await res.json();
      if (!res.ok || !j.ok) throw new Error(j.error || "index failed");
      indexStatus.textContent = `완료: 추가 ${j.added_chunks}개 / 총 ${j.total_chunks}개 (dim=${j.dim})`;
    } catch (e) {
      indexStatus.textContent = `오류: ${e}`;
    }
  });

  btnClear?.addEventListener("click", async () => {
    const collection_id = (collectionEl.value || "default").trim();
    if (!confirm(`'${collection_id}' 인덱스를 삭제할까요?`)) return;
    try {
      const res = await fetch("/api/rag/clear", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ collection_id })
      });
      const j = await res.json();
      if (!res.ok || !j.ok) throw new Error(j.error || "clear failed");
      indexStatus.textContent = `인덱스 삭제 완료 (${j.cleared})`;
    } catch (e) {
      indexStatus.textContent = `오류: ${e}`;
    }
  });

  // 질문
  async function ask() {
    const q = userInput.value.trim();
    if (!q) return;
    bubble("user", q);
    userInput.value = "";

    const load = loaderBubble();
    try {
      const res = await fetch("/api/rag/ask", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          collection_id: (collectionEl.value || "default").trim(),
          question: q,
          top_k: 4,
          temperature: 0.4,
          session_id: sessionId
        })
      });
      const j = await res.json();
      load.remove();

      if (!res.ok || !j.ok) {
        bubble("bot", j.error || "오류가 발생했습니다.");
        return;
      }
      bubble("bot", j.reply || "(응답 없음)");


    } catch (e) {
      load.remove();
      bubble("bot", `네트워크 오류: ${e}`);
    }
  }

  sendBtn?.addEventListener("click", ask);
  userInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  });
})();