document.addEventListener("DOMContentLoaded", function () {
  new TomSelect("#period", {
     create: false,               
     controlInput: false,         
     allowEmptyOption: true,       
     copyClassesToDropdown: true,
     dropdownParent: "body"
   });

  document.getElementById("periodSelect").addEventListener("change", function (e) {
    const selectedValue = e.target.value;
    console.log("선택된 기간:", selectedValue);

    updateMap(selectedValue);

    updateChart(selectedValue);
  });

  function updateMap(period) {

    const mapDiv = document.getElementById("fireMap");
    mapDiv.innerHTML = `📍 ${period} 기간의 산불 발생 지도 표시 중...`;
  }

  function updateChart(period) {
    const chartDiv = document.querySelector(".chart-placeholder");
    chartDiv.innerHTML = `📊 ${period} 기간의 산불 발생 추이 차트`;
  }
});

document.addEventListener("DOMContentLoaded", function () {
  new TomSelect("#period", {
    placeholder: "조회 기간 선택",
    allowEmptyOption: true
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const ctx = document.getElementById("yearlyFireChart");
  if (!ctx) {
    console.warn("⛔️ 차트용 canvas가 없음!");
    return;
  }

  new Chart(ctx, {
    type: "line",
    data: {
      labels: ["2019", "2020", "2021", "2022", "2023", "2024", "2025"],
      datasets: [{
        label: "산불 발생 건수",
        data: [12, 14, 18, 10, 20, 25, 17],
        borderColor: "#ff5722",
        backgroundColor: "rgba(255, 87, 34, 0.2)",
        tension: 0.4,
        fill: true,
        pointRadius: 5,
        pointBackgroundColor: "#ff5722"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => ` ${context.raw}건`
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "연도",
            font: { weight: 'bold' }
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "발생 건수",
            font: { weight: 'bold' }
          }
        }
      }
    }
  });
});