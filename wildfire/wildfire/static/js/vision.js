// 최소 업로드/결과 표시 전용
document.addEventListener("DOMContentLoaded", () => {
  const $ = (s) => document.querySelector(s);

  const form = $("#vf-form");
  const videoIn = $("#vf-video");
  const fileNameSpan = $("#vf-file-name");
  const inputPreview = $("#vf-input-preview");
  const outputPreview = $("#vf-output-preview");
  const statusBadge = $("#vf-status");
  const alertBadge = $("#vf-alert");
  const detTableBody = $("#vf-det-tbody");
  const downloadWrap = $("#vf-download-wrap");
  const downloadLink = $("#vf-download-link");
  const container = $(".vf-container");
  const endpointRun = container.dataset.endpointRun || "/vision/run";

  // 파일 선택 미리보기
  videoIn.addEventListener("change", () => {
    const f = videoIn.files?.[0];
    if (!f) {
      fileNameSpan.textContent = "";
      inputPreview.removeAttribute("src");
      inputPreview.load();
      return;
    }
    fileNameSpan.textContent = `선택된 파일: ${f.name} (${Math.round(f.size/1024/1024)}MB)`;
    const url = URL.createObjectURL(f);
    inputPreview.src = url;
    inputPreview.load();
  });

  // 상태/스피너
  let spinnerEl = null;
  function setStatus(text, type = "muted", spinning = false) {
    statusBadge.textContent = text;
    statusBadge.className = "vf-badge " + (type === "danger" ? "vf-badge-danger" : "vf-badge-muted");
    if (spinning) {
      if (!spinnerEl) {
        spinnerEl = document.createElement("span");
        spinnerEl.className = "vf-spinner";
        spinnerEl.setAttribute("aria-hidden", "true");
        statusBadge.prepend(spinnerEl);
        statusBadge.style.display = "inline-flex";
        statusBadge.style.alignItems = "center";
        statusBadge.style.gap = "6px";
      }
    } else {
      spinnerEl?.parentNode?.removeChild(spinnerEl);
      spinnerEl = null;
    }
  }
  function toggleAlert(on){ on ? alertBadge.classList.remove("vf-hidden") : alertBadge.classList.add("vf-hidden"); }
  function showToast(msg, ms=2500){
    let t = document.querySelector(".vf-toast");
    if(!t){ t = document.createElement("div"); t.className="vf-toast"; document.body.appendChild(t); }
    t.textContent = msg; t.classList.add("show"); setTimeout(()=>t.classList.remove("show"), ms);
  }
  function renderDetections(rows){
    detTableBody.innerHTML = "";
    if(!rows?.length) return;
    const frag = document.createDocumentFragment();
    rows.forEach(r=>{
      const tr=document.createElement("tr");
      const tdT=document.createElement("td"); tdT.textContent = r.t ?? r.time ?? r.sec ?? r.frame ?? "-";
      const tdL=document.createElement("td"); tdL.textContent = r.label ?? r.class ?? r.name ?? "-";
      const tdC=document.createElement("td"); const c=r.conf ?? r.confidence ?? r.score ?? r.probability;
      tdC.textContent = typeof c==="number"? c.toFixed(2) : (c ?? "-");
      tr.append(tdT,tdL,tdC); frag.appendChild(tr);
    });
    detTableBody.appendChild(frag);
  }

  // 제출
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = videoIn.files?.[0];
    if (!f) { showToast("동영상 파일을 선택해 주세요."); setStatus("파일 필요", "danger", false); return; }

    const formData = new FormData();
    formData.append("video_in", f);

    toggleAlert(false);
    renderDetections([]);
    outputPreview.removeAttribute("src");
    outputPreview.load();
    downloadWrap.classList.add("vf-hidden");
    setStatus("분석 중...", "muted", true);

    try{
      const res = await fetch(endpointRun, { method:"POST", body:formData });
      const data = await res.json();

      // 결과 반영
      if (data.processed_url) {
        outputPreview.src = data.processed_url;
        outputPreview.load();
        downloadLink.href = data.processed_url;
        downloadWrap.classList.remove("vf-hidden");
      }
      renderDetections(data.detections || []);
      toggleAlert((data.events||[]).length > 0 || (data.detections||[]).some(d => /fire|smoke/i.test(d.label||"")));
      setStatus(data.ok ? "완료" : "실패", data.ok ? "muted" : "danger", false);

      // 디버그(선택)
      if (data.log_tail) console.debug("[vision] backend log:\n" + data.log_tail.join("\n"));
    }catch(err){
      console.error(err);
      setStatus("실패", "danger", false);
      showToast("분석에 실패했습니다. 서버 로그를 확인하세요.");
    }
  });
});

