(() => {
  // ---- 유틸 ----
  const $ = (sel) => document.querySelector(sel);
  const log = (...a) => console.log("[RAG]", ...a);

  // 요소
  const fileInput   = $("#files");
  const btnIndex    = $("#btnIndex");
  const indexStatus = $("#indexStatus");
  const btnRefresh  = $("#btnRefreshStats") || $("#btnRefreshStats".replace('#','')); // 있으면 사용
  const btnReindex  = $("#btnReindex") || $("#btnReindex".replace('#',''));
  const btnRefreshFiles = $("#btnRefreshFiles");
  const corpusFilesList = $("#corpusFilesList");
  const queryBox    = $("#ragQuery");
  const topKInput   = $("#topK");
  const thrInput    = $("#threshold");
  const btnSearch   = $("#btnSearch");
  const resultsBox  = $("#results");
  const statCollection = $("#statCollection");
  const statPoints     = $("#statPoints");

  function setStatus(msg, kind="info") {
    if (!indexStatus) return;
    indexStatus.textContent = msg || "";
    indexStatus.style.color = (kind === "error" ? "#b00" : "#1b3f31");
  }

  async function refreshStats() {
    try {
      const r = await fetch("/api/rag/stats");
      const j = await r.json();
      if (j.ok) {
        if (statCollection) statCollection.textContent = j.stats.collection || "-";
        if (statPoints)     statPoints.textContent     = String(j.stats.points ?? "-");
      } else {
        setStatus(`상태 조회 실패: ${j.error}`, "error");
      }
    } catch (e) {
      setStatus(`상태 조회 오류: ${e}`, "error");
    }
  }

  async function refreshCorpusFiles() {
    if (!corpusFilesList) return;
    
    try {
      corpusFilesList.innerHTML = '<div class="muted small">파일 목록을 불러오는 중...</div>';
      
      const r = await fetch("/api/rag/corpus-files");
      const j = await r.json();
      
      if (j.ok) {
        if (j.files && j.files.length > 0) {
          corpusFilesList.innerHTML = j.files.map(file => {
            const date = new Date(file.modified * 1000).toLocaleString('ko-KR');
            const size = (file.size / 1024).toFixed(1) + 'KB';
            return `
              <div class="corpus-file-item">
                <div class="file-name">📄 ${file.name}</div>
                <div class="file-info muted small">${size} • ${date}</div>
              </div>
            `;
          }).join("");
        } else {
          corpusFilesList.innerHTML = '<div class="muted small">저장된 파일이 없습니다.</div>';
        }
      } else {
        corpusFilesList.innerHTML = `<div class="muted small">파일 목록 조회 실패: ${j.error}</div>`;
      }
    } catch (e) {
      corpusFilesList.innerHTML = `<div class="muted small">파일 목록 조회 오류: ${e}</div>`;
    }
  }

  async function handleIndex() {
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
      setStatus("업로드할 파일을 선택하세요.", "error");
      return;
    }
    
    const fileCount = fileInput.files.length;
    const fd = new FormData();
    [...fileInput.files].forEach(f => fd.append("files", f));
    
    setStatus(`📥 ${fileCount}개 파일 업로드 및 처리 중...`);
    log("POST /api/rag/index-files", fileCount);

    try {
      const res = await fetch("/api/rag/index-files", { method: "POST", body: fd });
      const data = await res.json();
      log("index-files response:", data);
      
      if (data.ok) {
        let statusMsg = `✅ 처리 완료: ${data.added} 청크 추가됨`;
        
        if (data.files && data.files.length > 0) {
          statusMsg += `\n\n📁 저장된 파일들:`;
          data.files.forEach(file => {
            statusMsg += `\n• ${file.original_name} → ${file.saved_name} (${file.chunks} 청크)`;
          });
          statusMsg += `\n\n💾 저장 위치: ${data.corpus_path}`;
        }
        
        setStatus(statusMsg);
        await refreshStats();
        await refreshCorpusFiles();
        
        // 파일 입력 필드 초기화
        if (fileInput) fileInput.value = '';
        
      } else {
        setStatus(`❌ 처리 실패: ${data.error || res.status}`, "error");
      }
    } catch (e) {
      setStatus(`❌ 네트워크 오류: ${e}`, "error");
    }
  }

  async function handleSearch() {
    const q = (queryBox?.value || "").trim();
    if (!q) {
      setStatus("검색어를 입력하세요.", "error");
      return;
    }
    const k = parseInt(topKInput?.value || "5", 10);
    const thr = parseFloat(thrInput?.value || "0.25");
    setStatus("🔎 검색 중…");
    resultsBox && (resultsBox.innerHTML = "");

    const url = `/api/rag/search?q=${encodeURIComponent(q)}&k=${k}&thr=${thr}`;
    log("GET", url);
    try {
      const r = await fetch(url);
      const j = await r.json();
      log("search response:", j);
      if (!j.ok) {
        setStatus(`❌ 검색 실패: ${j.error}`, "error");
        return;
      }
      setStatus("✅ 검색 완료");
      if (resultsBox) {
        if (!j.results || j.results.length === 0) {
          resultsBox.innerHTML = '<div class="muted small">검색 결과가 없습니다.</div>';
          return;
        }
        resultsBox.innerHTML = j.results.map((h, idx) => `
          <div class="result-item">
            <div><strong>#${idx+1}</strong> <span class="score">score: ${h.score?.toFixed?.(3) ?? h.score}</span></div>
            <div class="meta small">source: ${h.meta?.filename || h.meta?.source || "-"}</div>
            <div style="white-space:pre-wrap;margin-top:8px">${(h.text || "").replace(/</g,"&lt;")}</div>
          </div>
        `).join("");
      }
    } catch (e) {
      setStatus(`❌ 검색 오류: ${e}`, "error");
    }
  }

  // 이벤트 바인딩
  document.addEventListener("DOMContentLoaded", () => {
    log("rag.js loaded, binding events…");
    btnIndex   && btnIndex.addEventListener("click", handleIndex);
    btnSearch  && btnSearch.addEventListener("click", handleSearch);
    btnRefresh && btnRefresh.addEventListener("click", refreshStats);
    btnRefreshFiles && btnRefreshFiles.addEventListener("click", refreshCorpusFiles);
    btnReindex && btnReindex.addEventListener("click", async () => {
      if (!confirm("인덱스를 전체 재생성할까요? 기존 내용이 초기화됩니다.")) return;
      setStatus("♻️ 전체 재생성 중…");
      try {
        const r = await fetch("/api/rag/reindex", { method:"POST" });
        const j = await r.json();
        log("reindex response:", j);
        if (j.ok) {
          setStatus(`✅ 재생성 완료: ${j.indexed} 청크`);
          await refreshStats();
        } else {
          setStatus(`❌ 재생성 실패: ${j.error}`, "error");
        }
      } catch (e) {
        setStatus(`❌ 재생성 오류: ${e}`, "error");
      }
    });

    // 첫 진입 시 상태
    refreshStats();
    refreshCorpusFiles();
  });
})();