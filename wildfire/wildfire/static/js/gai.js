document.addEventListener("DOMContentLoaded", () => {
  const chatWindow = document.getElementById("chatWindow");
  const userInput  = document.getElementById("userInput");
  const sendBtn    = document.getElementById("sendBtn");
  const resetBtn   = document.getElementById("resetChat");

  let isSending = false;

  function scrollToBottom() {
    if (chatWindow) chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function appendUserMessage(text) {
    if (!chatWindow) return;
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
    if (!chatWindow) return;
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

  function appendBotLoading() {
    if (!chatWindow) return { stopLoading: () => {} };
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
    row._interval = setInterval(() => {
      img.src = toggle ? "/static/img/bot02.png" : "/static/img/bot01.png";
      toggle = !toggle;
    }, 450);

    row.stopLoading = () => clearInterval(row._interval);
    return row;
  }

  function setSendingState(sending) {
    isSending = sending;
    if (sendBtn)   sendBtn.disabled = sending;
    if (userInput) userInput.disabled = sending;
  }

  async function sendMessage() {
    if (!userInput) return;
    const text = userInput.value.trim();
    if (!text || isSending) return;

    appendUserMessage(text);
    userInput.value = "";

    const loader = appendBotLoading();
    setSendingState(true);

    try {
      const res = await fetch("/api/gai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })  // 백엔드에서 자동 분기
      });

      const data = await res.json().catch(() => ({}));
      if (loader && loader.stopLoading) loader.stopLoading();

      if (!res.ok || !data.ok) {
        appendBotMessage(data.error || "오류가 발생했습니다.");
        return;
      }

      appendBotMessage(data.reply || "");
    } catch (e) {
      if (loader && loader.stopLoading) loader.stopLoading();
      appendBotMessage("네트워크 오류가 발생했습니다.");
    } finally {
      setSendingState(false);
      userInput.focus();
    }
  }

  function resetChat() {
    if (chatWindow) chatWindow.innerHTML = "";
    if (userInput) { userInput.value = ""; userInput.focus(); }
  }

  if (sendBtn) sendBtn.addEventListener("click", sendMessage);
  if (userInput) {
    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }
  if (resetBtn) resetBtn.addEventListener("click", resetChat);

  if (userInput) userInput.focus();
});