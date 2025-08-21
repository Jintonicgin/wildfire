const chatWindow = document.getElementById("chatWindow");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const resetBtn = document.getElementById("resetChat");
const modeSelect = document.getElementById("modeSelect");
const tempInput = document.getElementById("temperature");

let isSending = false;
let messages = [];

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
  img.src = "/static/img/bot1.png";
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
  const row = document.createElement("div");
  row.className = "chat-message bot";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  const img = document.createElement("img");
  img.src = "/static/img/bot1.png";
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
    img.src = toggle ? "/static/img/bot2.png" : "/static/img/bot1.png";
    toggle = !toggle;
  }, 500);
  row.stopLoading = () => clearInterval(row._interval);
  return row;
}

function setSendingState(sending) {
  isSending = sending;
  sendBtn.disabled = sending;
  userInput.disabled = sending;
  modeSelect.disabled = sending;
  tempInput.disabled = sending;
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isSending) return;

  const mode = modeSelect.value || "chat";
  const temperature = parseFloat(tempInput.value || "0.7");

  appendUserMessage(text);
  messages.push({ role: "user", content: text });
  userInput.value = "";

  const loader = appendBotLoading();
  setSendingState(true);

  try {
    const res = await fetch("/api/gai/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        temperature,
        messages
      })
    });

    const data = await res.json();
    loader.stopLoading();

    if (!res.ok || !data.ok) {
      appendBotMessage(data.error || "오류가 발생했습니다.");
      return;
    }

    const reply = data.reply || "";
    messages.push({ role: "assistant", content: reply });
    appendBotMessage(reply);
  } catch (e) {
    loader.stopLoading();
    appendBotMessage("네트워크 오류가 발생했습니다.");
  } finally {
    setSendingState(false);
    userInput.focus();
  }
}

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