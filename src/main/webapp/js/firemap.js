window.kakao.maps.load(function () {
  const container = document.getElementById("fireMap");

  const map = new kakao.maps.Map(container, {
    center: new kakao.maps.LatLng(37.8228, 128.1555),
    level: 9,
  });

  const initialBounds = new kakao.maps.LatLngBounds(
    new kakao.maps.LatLng(37.0, 127.5),
    new kakao.maps.LatLng(38.5, 129.5)
  );
  map.setBounds(initialBounds);

  const fireData = [
    {
      name: "강릉시 왕산면",
      date: "2025-06-30",
      area: "12ha",
      cause: "입산자 실화",
      lat: 37.7535,
      lng: 128.8746,
    },
    {
      name: "평창군 봉평면",
      date: "2025-06-29",
      area: "5ha",
      cause: "번개",
      lat: 37.5912,
      lng: 128.3897,
    },
    {
      name: "정선군 고한읍",
      date: "2025-06-28",
      area: "8.5ha",
      cause: "건조한 날씨",
      lat: 37.1994,
      lng: 128.8361,
    },
  ];

  let currentMarkers = [];
  let currentOverlays = [];

  function clearMap() {
    currentMarkers.forEach((m) => m.setMap(null));
    currentOverlays.forEach((o) => o.setMap(null));
    currentMarkers = [];
    currentOverlays = [];
  }

  function filterFiresByPeriod(value) {
    const today = new Date("2025-06-30");
    if (value === "today") {
      return fireData.filter(f => f.date === "2025-06-30");
    }

    const diff = {
      "3d": 3,
      "7d": 7
    }[value];

    if (!diff) return [];

    return fireData.filter((f) => {
      const fireDate = new Date(f.date);
      const delta = (today - fireDate) / (1000 * 60 * 60 * 24);
      return delta >= 0 && delta < diff;
    });
  }

  function showFiresOnMap(filteredFires) {
    clearMap();

    filteredFires.forEach((fire) => {
      const position = new kakao.maps.LatLng(fire.lat, fire.lng);

      const marker = new kakao.maps.Marker({
        map: map,
        position: position,
        title: fire.name,
      });
      currentMarkers.push(marker);

      const content = `
        <div style="padding:12px; background:#fff; border:1px solid #aaa; border-radius:12px; 
                    box-shadow: 2px 3px 6px rgba(0,0,0,0.2); font-size:15px; font-family:'Noto Sans KR'">
          <strong style="font-size:17px;">${fire.name}</strong><br>
          날짜: <span style="color:#cc0000;">${fire.date}</span><br>
          면적: <span style="color:#cc0000;">${fire.area}</span><br>
          원인: <span style="color:#cc0000;">${fire.cause}</span>
        </div>
      `;

      const overlay = new kakao.maps.CustomOverlay({
        content: content,
        position: position,
        yAnchor: 1.3,
      });

      overlay.setMap(map);
      currentOverlays.push(overlay);
    });
  }

  document.getElementById("periodConfirmBtn").addEventListener("click", () => {
    const selected = document.getElementById("period").value;
    if (!selected) {
      clearMap();
      return;
    }

    const filtered = filterFiresByPeriod(selected);
    showFiresOnMap(filtered);
  });

  document.getElementById("period").addEventListener("change", () => {
    clearMap();
  });
});