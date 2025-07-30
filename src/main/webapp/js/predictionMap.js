let geojsonData = null;
let polygonOverlay = null;

fetch("/WildFire/data/gangwondo_ssg.geojson")
  .then((res) => res.json())
  .then((data) => {
    geojsonData = data;
  })
  .catch((err) => {
    console.error("GeoJSON 로드 오류:", err);
    alert("경계 데이터를 불러오지 못했습니다.");
  });

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
    alert("좌표 파싱 오류 발생");
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
    strokeStyle: "solid",
  });

  polygonOverlay.setMap(map); 

  const bounds = new kakao.maps.LatLngBounds();
  paths.forEach((path) => path.forEach((point) => bounds.extend(point)));
  map.setBounds(bounds);

  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();
  const centerLat = (sw.getLat() + ne.getLat()) / 2;
  const centerLng = (sw.getLng() + ne.getLng()) / 2;

  const latInput = document.getElementById('latitude');
  const lngInput = document.getElementById('longitude');
  if (latInput && lngInput) {
    latInput.value = centerLat.toFixed(6);
    lngInput.value = centerLng.toFixed(6);
  }
});