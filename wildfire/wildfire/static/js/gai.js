  const chatWindow = document.getElementById("chatWindow");
  const userInput  = document.getElementById("userInput");
  const sendBtn    = document.getElementById("sendBtn");
  const resetBtn   = document.getElementById("resetChat");

  let isSending = false;
  let messages = [];

  // --- 세션 고정: 항상 동일한 session_id 사용 ---
let sessionId = localStorage.getItem("gai_session_id");
if (!sessionId) {
  sessionId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now());
  localStorage.setItem("gai_session_id", sessionId);
}

  function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "chat-message user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  function appendBotMessage(text) {
    const row = document.createElement("div");
    row.className = "chat-message bot";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    const img = document.createElement("img");
    img.src = "/static/img/bot01.png";
    avatar.appendChild(img);

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatWindow.appendChild(row);
    scrollToBottom();
  }

  // 로딩버블 (깜빡임 포함)
  function appendBotLoading() {
    const row = document.createElement("div");
    row.className = "chat-message bot";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    const img = document.createElement("img");
    img.src = "/static/img/bot01.png";
    avatar.appendChild(img);

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = "생성 중…";

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatWindow.appendChild(row);
    scrollToBottom();

    let toggle = true;
    const iv = setInterval(() => {
      img.src = toggle ? "/static/img/bot02.png" : "/static/img/bot01.png";
      toggle = !toggle;
    }, 450);

    row.stopLoading = () => clearInterval(iv);
    row.removeLoading = () => { clearInterval(iv); row.remove(); };

    return row;
  }

  function setSendingState(sending) {
    isSending = sending;
    sendBtn.disabled = sending;
    userInput.disabled = sending;
  }

  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isSending) return;

    appendUserMessage(text);
    messages.push({ role: "user", content: text });
    userInput.value = "";

    const loader = appendBotLoading();
    setSendingState(true);

    try {
      const res = await fetch("/api/gai/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: sessionId })
    });

      const data = await res.json().catch(() => ({}));

      loader.removeLoading();

      if (!res.ok || !data.ok) {
        appendBotMessage(data.error || "오류가 발생했습니다.");
        return;
      }

      const reply = data.reply ?? "";
      messages.push({ role: "assistant", content: reply });
      appendBotMessage(reply);

      // 서버가 새 session_id를 회신하면(보통은 같음) 갱신
      if (data.session_id && data.session_id !== sessionId) {
    sessionId = data.session_id;
    localStorage.setItem("gai_session_id", sessionId);
  }
    } catch (e) {
      loader.removeLoading();
      appendBotMessage("네트워크 오류가 발생했습니다.");
    } finally {
      setSendingState(false);
      userInput.focus();
    }
  }

  // 대화만 초기화 (세션 유지)
  function resetChat() {
    messages = [];
    chatWindow.innerHTML = "";
    userInput.value = "";
    userInput.focus();
  }

  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  resetBtn.addEventListener("click", resetChat);

  userInput.focus();
