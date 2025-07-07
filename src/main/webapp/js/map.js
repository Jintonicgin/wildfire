window.kakao.maps.load(function () {
  const container = document.getElementById("map");
  const map = new kakao.maps.Map(container, {
    center: new kakao.maps.LatLng(37.8228, 128.1555),
    level: 9,
  });

  let geojsonData = null;
  let polygonOverlay = null;
  let infoOverlay = null;

  const initialBounds = new kakao.maps.LatLngBounds(
    new kakao.maps.LatLng(37.0, 127.5),
    new kakao.maps.LatLng(38.5, 129.5)
  );
  map.setBounds(initialBounds);

  const recoveryStats = {
    "강릉시": {
      recoveryRate: "84.7%",
      damageArea: "15.6㎢",
      ndviBefore: 0.65,
      ndviAfter: 0.48,
      ndviAfter2Yrs: 0.72,
    },
  };

  fetch("/WildFire/data/gangwondo_ssg.geojson")
    .then((res) => res.json())
    .then((data) => {
      geojsonData = data;
    })

  document.getElementById("confirmBtn").addEventListener("click", () => {
    const selectedCity = document.getElementById("city").value.trim();

    if (!geojsonData || !selectedCity) {
      alert("데이터 로드 또는 선택값 오류");
      return;
    }

    if (polygonOverlay) {
      polygonOverlay.setMap(null);
      polygonOverlay = null;
    }
    if (infoOverlay) {
      infoOverlay.setMap(null);
      infoOverlay = null;
    }

    const foundFeature = geojsonData.features.find(
      (f) => f.properties.title?.trim() === selectedCity
    );

    if (!foundFeature) {
      alert("해당 시군구를 찾을 수 없습니다.");
      return;
    }

    const { type, coordinates } = foundFeature.geometry;
    const paths = [];

    try {
      if (type === "Polygon") {
        const validPath = coordinates[0].map(([x, y]) =>
          new kakao.maps.LatLng(y, x)
        );
        if (validPath.length) paths.push(validPath);
      } else if (type === "MultiPolygon") {
        coordinates.forEach((poly) => {
          const validPath = poly[0].map(([x, y]) =>
            new kakao.maps.LatLng(y, x)
          );
          if (validPath.length) paths.push(validPath);
        });
      }
    } catch (e) {
      console.error("좌표 변환 오류:", e);
      return;
    }

    if (!paths.length) {
      alert("경계 좌표 없음");
      return;
    }

    polygonOverlay = new kakao.maps.Polygon({
      path: paths,
      strokeWeight: 2,
      strokeColor: "#2E8B57",
      strokeOpacity: 0.9,
      fillColor: "#a1d1a1",
      fillOpacity: 0.5,
    });

	
    polygonOverlay.setMap(map);

    const bounds = new kakao.maps.LatLngBounds();
    paths.forEach((path) => path.forEach((p) => bounds.extend(p)));
    map.setBounds(bounds);
	
    function getPolygonCentroid(path) {
      let sumLat = 0,
        sumLng = 0;
      path.forEach((point) => {
        sumLat += point.getLat();
        sumLng += point.getLng();
      });
      return new kakao.maps.LatLng(sumLat / path.length, sumLng / path.length);
    }

    const centroid = getPolygonCentroid(paths[0]);
    const stats = recoveryStats[selectedCity];

    if (stats) {
		const infoHtml = `
		  <div class="region-info-box">
		    <strong>${selectedCity}</strong>
		    <div>복원율: ${stats.recoveryRate}</div>
		    <div>피해면적: ${stats.damageArea}</div>
		    <div>직전 NDVI: ${stats.ndviBefore}</div>
		    <div>직후 NDVI: ${stats.ndviAfter}</div>
		    <div>2년 후 NDVI: ${stats.ndviAfter2Yrs}</div>
		  </div>
		`;
			
	  
      infoOverlay = new kakao.maps.CustomOverlay({
        content: infoHtml,
        position: centroid,
		xAnchor: 0.5, 
        yAnchor: 0.1,
      });
      infoOverlay.setMap(map);
    }
  });
});