document.addEventListener("DOMContentLoaded", async function () {
  const ndviCtx = document.getElementById("ndviChart").getContext("2d");
  const barCtx = document.getElementById("barChart").getContext("2d");
  let ndviChart, barChart;

  // ✅ NDVI 차트
  try {
    const response = await fetch("/WildFire/data/ndvi_yearly_gangwon.csv");
    const csvText = await response.text();
    const rows = csvText.trim().split("\n").slice(1);
    const labels = [], values = [];

    rows.forEach(row => {
      const [_, ndvi, year] = row.split(",");
      labels.push(year.trim());
      values.push(parseFloat(ndvi).toFixed(4));
    });

    ndviChart = new Chart(ndviCtx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "NDVI 값",
          data: values,
          borderColor: '#007f5f',
          backgroundColor: 'rgba(0,127,95,0.2)',
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointHitRadius: 10,
          pointBackgroundColor: '#007f5f'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1500 },
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#fff',
            borderColor: '#007f5f',
            borderWidth: 1,
            titleColor: '#007f5f',
            bodyColor: '#2e3b3a',
            callbacks: {
              label: ctx => `연도: ${ctx.label}, NDVI: ${parseFloat(ctx.raw).toFixed(4)}`
            }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 1.0,
            title: { display: true, text: "NDVI 값" },
            ticks: { stepSize: 0.02 }
          },
          x: {
            title: { display: true, text: "연도" },
            ticks: { autoSkip: false, maxRotation: 0, minRotation: 0 }
          }
        }
      }
    });
  } catch (err) {
    console.error("NDVI 차트 로딩 실패:", err);
  }

  // ✅ 변수 영향력 + Top3 변수
  try {
    const res = await fetch("/WildFire/data/importance.json");
    const json = await res.json();

    const labelMap = {
      mean_temp: "기온",
      year: "연도",
      total_precip: "강수량 총합",
      mean_pressure: "기압",
      mean_wind: "풍속"
    };

    const iconMap = {
      mean_temp: "/WildFire/img/icon_temp.png",
      year: "/WildFire/img/icon_year.png",
      total_precip: "/WildFire/img/icon_precipitation.png",
      mean_pressure: "/WildFire/img/icon_pressure.png",
      mean_wind: "/WildFire/img/icon_wind.png"
    };

    const labels = Object.keys(json).map(k => labelMap[k] || k);
    const values = Object.values(json);

    // 🔷 Bar Chart
    barChart = new Chart(barCtx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "변수 영향력(%)",
          data: values,
          backgroundColor: ['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0'],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1000 },
        interaction: {
          mode: 'index',
          intersect: false,
          axis: 'x'
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#fff',
            borderColor: '#4CAF50',
            borderWidth: 1,
            titleColor: '#4CAF50',
            bodyColor: '#2e3b3a',
            callbacks: {
              label: ctx => `${ctx.label}: ${ctx.raw.toFixed(2)}%`
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: { display: true, text: "영향력 (%)" }
          }
        }
      }
    });

    // 🔷 Top3 박스
    const top3 = Object.entries(json).sort((a, b) => b[1] - a[1]).slice(0, 3);
    const top3Box = document.getElementById("top3-variables");
    top3Box.innerHTML = "";

    top3.forEach(([key]) => {
      const div = document.createElement("div");
      div.className = "top3-variable";
      div.innerHTML = `
        <img src="${iconMap[key] || "/WildFire/img/default_icon.png"}" alt="${labelMap[key]} 아이콘" />
        <span>${labelMap[key] || key}</span>
      `;
      top3Box.appendChild(div);
    });

  } catch (err) {
    console.error("변수 영향력 or Top3 변수 로딩 실패:", err);
  }

  // ✅ 산림현황 슬라이드: 기본 강원도
  await loadForestSlides("강원도");

  // ✅ 지역 선택 시 갱신
  document.getElementById("confirmBtn")?.addEventListener("click", () => {
    const city = document.getElementById("city")?.value.trim();
    if (!city) return;

    updateTitlesByRegion(city);
    updateNdviChartForRegion();
    loadForestSlides(city); // 🔁 슬라이드 갱신
    if (typeof drawSelectedRegion === "function") drawSelectedRegion(city);
  });
});


// 🔷 슬라이드 렌더링 함수
async function loadForestSlides(region = "강원도") {
  try {
    const res = await fetch("/WildFire/data/gangwon_forest_summary.json");
    const data = await res.json();

    const summary = data[region];
    if (!summary) {
      console.warn(`⛔ ${region} 데이터 없음`);
      return;
    }

    const items = [
      { title: "산림 면적", value: summary["산림면적"] },
      { title: "임목 축적", value: summary["임목축적"] },
      { title: "ha당 임목 축적", value: summary["ha당임목축적"] },
      { title: "산림율", value: summary["산림율"] }
    ];

    const container = document.querySelector(".forest-slide-list");
    container.innerHTML = "";

    items.forEach((item, i) => {
      const slide = document.createElement("div");
      slide.className = `forest-slide${i === 0 ? " active" : ""}`;
      slide.innerHTML = `
        <h5>${item.title}</h5>
        <p><strong>${item.value}</strong></p>
        <p class="year">2020년 기준</p>
      `;
      container.appendChild(slide);
    });

    const slides = document.querySelectorAll(".forest-slide");
    let currentIndex = 0;

    function updateSlides() {
      slides.forEach(slide => slide.classList.remove("active"));
      slides[currentIndex].classList.add("active");
    }

    document.getElementById("forest-prev")?.addEventListener("click", () => {
      currentIndex = (currentIndex - 1 + slides.length) % slides.length;
      updateSlides();
    });

    document.getElementById("forest-next")?.addEventListener("click", () => {
      currentIndex = (currentIndex + 1) % slides.length;
      updateSlides();
    });

    updateSlides();
  } catch (err) {
    console.error("산림현황 슬라이드 로딩 실패:", err);
  }
}


// 🔷 제목 변경
function updateTitlesByRegion(regionName) {
  document.getElementById("map-title").textContent = `${regionName} 복원력 지도`;
  document.getElementById("forest-title").textContent = `${regionName} 산림 현황`;
  document.getElementById("ndvi-title").textContent = `${regionName}의 산불 이후 NDVI 변화율`;
}

// 🔷 NDVI 변화율 예시
function updateNdviChartForRegion() {
  const ndviChart = Chart.getChart("ndviChart");
  if (!ndviChart) return;
  ndviChart.data.labels = ["산불 직전", "산불 직후", "1개월 후", "3개월 후", "6개월 후", "12개월 후", "18개월 후", "24개월 후"];
  ndviChart.data.datasets[0].data = [0.82, 0.55, 0.60, 0.68, 0.72, 0.74, 0.76, 0.78];
  ndviChart.update();
}

  function saveCsvToSessionFromCharts() {
    const csvArray = [];
    const ndvi = Chart.getChart("ndviChart");
    const bar = Chart.getChart("barChart");

    if (ndvi) {
      csvArray.push(["[NDVI 변화율]"], ["시점", "NDVI"]);
      ndvi.data.labels.forEach((label, i) => {
        csvArray.push([label, ndvi.data.datasets[0].data[i]]);
      });
      csvArray.push([""]);
    }

    if (bar) {
      csvArray.push(["[변수 영향력]"], ["변수", "영향력(%)"]);
      bar.data.labels.forEach((label, i) => {
        csvArray.push([label, bar.data.datasets[0].data[i]]);
      });
      csvArray.push([""]);
    }

    csvArray.push(["[산림 현황 카드]"], ["항목", "값", "기준년도"]);
    document.querySelectorAll(".forest-slide").forEach(slide => {
      const title = slide.querySelector("h5")?.textContent || "";
      const value = slide.querySelector("strong")?.textContent || "";
      const year = slide.querySelector(".year")?.textContent || "";
      csvArray.push([title, value, year]);
    });

    fetch("/WildFire/StoreCsvServlet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(csvArray)
    }).then(res => {
      if (res.ok) {
        console.log("csvData 저장 성공");
      } else {
        console.error("csvData 저장 실패", res.status);
      }
    });
  }

  const downloadLinks = document.querySelectorAll(".download-link");
  downloadLinks.forEach(link => {
    link.addEventListener("click", function (e) {
      if (!window.isLoggedIn) {
        e.preventDefault();
        alert("로그인 후 이용하실 수 있습니다.");
        window.location.href = "/WildFire/jsp/login.jsp";
        return;
      }

      if (this.dataset.type === "pdf") {
        e.preventDefault();
        sendChartImagesToPDFExport();
      }
    });
  });

  function sendChartImagesToPDFExport() {
    const region = document.getElementById("city")?.value.trim();
    const ndviCanvas = document.getElementById("ndviChart");
    const barCanvas = document.getElementById("barChart");

    const ndviStart = ndviChart?.data?.datasets[0]?.data[1] ?? 0;
    const ndviEnd = ndviChart?.data?.datasets[0]?.data[7] ?? 0;
    const varLabels = barChart?.data?.labels ?? [];
    const varValues = barChart?.data?.datasets[0]?.data ?? [];

    fetch("/WildFire/DownloadPDFServlet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        region,
        ndvi: ndviCanvas.toDataURL("image/png"),
        bar: barCanvas.toDataURL("image/png"),
        ndviStart: ndviStart.toFixed(3),
        ndviEnd: ndviEnd.toFixed(3),
        varLabels,
        varValues
      })
    })
      .then(res => res.blob())
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "ndvi_report.pdf";
        a.click();
      })
      .catch(err => {
        console.error("PDF 생성 중 오류:", err);
        alert("PDF 생성에 실패했습니다.");
      });
  }

  function toggleMobileMenu() {
    document.getElementById("mobile-menu").style.right = "0px";
    document.getElementById("mobile-overlay").style.display = "block";
  }

  function closeMobileMenu() {
    document.getElementById("mobile-menu").style.right = "-240px";
    document.getElementById("mobile-overlay").style.display = "none";
  }
