
let map;
let currentPolygon = null;
let currentArrow = null;
let currentLabel = null;
let infoOverlay = null;

document.addEventListener("DOMContentLoaded", function () {
    // TomSelect 인스턴스를 변수에 저장
    const hourSelect = new TomSelect("#forecastHour", {
        create: false,
        allowEmptyOption: true,
        controlInput: false,
        sortField: [],
        maxOptions: 3,
        placeholder: "예측 시간 선택"
    });
    const citySelect = new TomSelect("#city", {
        create: false,
        allowEmptyOption: true,
        controlInput: false,
        sortField: [],
        maxOptions: 19,
        placeholder: "시/군 선택"
    });

    kakao.maps.load(function () {
        const mapContainer = document.getElementById('predictionMap');
        map = new kakao.maps.Map(mapContainer, {
            center: new kakao.maps.LatLng(37.8228, 128.1555),
            level: 9
        });

        const initialBounds = new kakao.maps.LatLngBounds(
            new kakao.maps.LatLng(37.0, 127.5),
            new kakao.maps.LatLng(38.5, 129.5)
        );
        map.setBounds(initialBounds);

        let marker = null;

        kakao.maps.event.addListener(map, 'click', function (mouseEvent) {
            const latlng = mouseEvent.latLng;
            const lat = latlng.getLat();
            const lng = latlng.getLng();

            if (marker) marker.setMap(null);
            marker = new kakao.maps.Marker({ position: latlng });
            marker.setMap(map);

            document.getElementById('latitude').value = lat.toFixed(6);
            document.getElementById('longitude').value = lng.toFixed(6);
            
            citySelect.clear();
        });
    });

    document.getElementById('predictionForm').addEventListener('submit', function (e) {
        e.preventDefault();

        if (!window.isLoggedIn) {
            alert("로그인 후 이용하실 수 있습니다.");
            window.location.href = "/WildFire/jsp/login.jsp";
            return;
        }

        const region = citySelect.getValue();
        const lat = parseFloat(document.getElementById('latitude').value);
        const lng = parseFloat(document.getElementById('longitude').value);
        const hour = parseInt(hourSelect.getValue());

        if (isNaN(hour)) {
            alert("예측 시간을 선택해주세요.");
            return;
        }

        const now = new Date();
        const future = new Date(now.getTime() + hour * 3600 * 1000);
        const timestamp = future.toISOString().split('.')[0];

        let inputData = {
            timestamp: timestamp,
            durationHours: hour
        };

        if (region) {
            inputData.city_name = region;
        } else if (!isNaN(lat) && !isNaN(lng)) {
            inputData.latitude = lat;
            inputData.longitude = lng;
        } else {
            alert("지역을 선택하거나 지도에서 위치를 클릭해주세요.");
            return;
        }

        showLoading(true);

        fetch('/WildFire/PredictionServlet', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        })
        .then(res => res.text())
        .then(text => {
            showLoading(false);

            try {
                const data = JSON.parse(text);

                if (data.error) {
                    alert("❌ 예측 실패: " + data.error);
                    console.error(data.details || data.error);
                    return;
                }
                
                const startLat = region ? data.path_trace[0].lat : lat;
                const startLng = region ? data.path_trace[0].lon : lng;

                document.getElementById('predictedDamageArea').textContent =
                    data.final_damage_area.toFixed(2) + " ha";

                const lastPathTrace = data.path_trace[data.path_trace.length - 1];
                const direction = convertDegreeToDirection(lastPathTrace.wind_direction_deg);

                let totalDistance;
          
                if (region) {
           
                    const area_m2 = data.final_damage_area * 10000;
                    const aspectRatio = 0.6; 
                    const b = Math.sqrt(area_m2 / (Math.PI * aspectRatio));
                    const a = b / aspectRatio; 
                    totalDistance = a;
                } else {
                    totalDistance = calculateDistance(startLat, startLng, data.final_lat, data.final_lon);
                }

                const speedValue = totalDistance / hour;
                const speedCategory = classifySpeed(speedValue);

                const directionContainer = document.getElementById('predictedSpreadDirection');
                directionContainer.innerHTML = ''; 
                const arrow = document.createElement('div');
                arrow.classList.add('direction-arrow', 'dir-' + direction);
                directionContainer.appendChild(arrow);

                const infoText = document.createElement('div');
                infoText.innerHTML = `
                    확산 방향: ${direction}<br/>
                    확산 거리: ${totalDistance.toFixed(2)} m<br/>
                    확산 속도: ${speedValue.toFixed(2)} m/h (${speedCategory})
                `;
                infoText.style.marginTop = '5px';
                directionContainer.appendChild(infoText);

                drawDamageEllipse(map, startLat, startLng, totalDistance, hour, direction, data.final_damage_area);
                addDirectionArrowOnMap(map, startLat, startLng, direction);
                addPredictionLabelOnMap(map, startLat, startLng, data.final_damage_area, direction, totalDistance, speedValue, speedCategory);

            } catch (e) {
                console.error("❌ JSON 파싱 오류:", e);
                console.error("서버 응답 원문:", text);
                alert("서버 응답 파싱에 실패했습니다.");
            }
        })
        .catch(err => {
            showLoading(false);
            console.error('❌ 요청 오류:', err);
            alert('서버 요청 중 문제가 발생했습니다.');
        });
    });

    function showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.toggle('hidden', !show);
        }
    }

    function calculateDistance(lat1, lon1, lat2, lon2) {
        if (lat1 === -999 || lon1 === -999) return 0; // DB 데이터에 위경도 없을 경우
        const R = 6371e3;
        const φ1 = lat1 * Math.PI / 180;
        const φ2 = lat2 * Math.PI / 180;
        const Δφ = (lat2 - lat1) * Math.PI / 180;
        const Δλ = (lon2 - lon1) * Math.PI / 180;

        const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
                  Math.cos(φ1) * Math.cos(φ2) *
                  Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    function convertDegreeToDirection(deg) {
        if (deg === -999 || deg === null || deg === undefined || isNaN(deg)) return 'N';
        const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const index = Math.floor(((deg + 22.5) % 360) / 45);
        return dirs[index];
    }

    function classifySpeed(speed) {
        if (speed < 100) return "느림";
        if (speed < 300) return "보통";
        return "빠름";
    }

    function drawDamageEllipse(map, startLat, startLng, speed, durationHours, direction, damageArea) {
        if (currentPolygon) currentPolygon.setMap(null);
        if (infoOverlay) infoOverlay.setMap(null);
        if (startLat === -999 || startLng === -999) return; 
        const directionAngles = { N: 0, NE: 45, E: 90, SE: 135, S: 180, SW: 225, W: 270, NW: 315 };
        const angle = (directionAngles[direction] || 0) * Math.PI / 180;

        const area_m2 = damageArea * 10000;
        const aspectRatio = 0.6;
        const b = Math.sqrt(area_m2 / (Math.PI * aspectRatio));
        const a = b / aspectRatio;

        const R = 6371000;
        const dx = (a / 2) * Math.cos(angle);
        const dy = (b / 2) * Math.sin(angle);

        const dLat = dy / R;
        const dLng = dx / (R * Math.cos(startLat * Math.PI / 180));

        const centerLat = startLat + dLat * (180 / Math.PI);
        const centerLng = startLng + dLng * (180 / Math.PI);

        const numPoints = 60;
        const coords = [];

        for (let i = 0; i < numPoints; i++) {
            const theta = (i / numPoints) * 2 * Math.PI;
            const x = a * Math.cos(theta);
            const y = b * Math.sin(theta);

            const xr = x * Math.cos(angle) - y * Math.sin(angle);
            const yr = x * Math.sin(angle) + y * Math.cos(angle);

            const dLat_point = yr / R;
            const dLng_point = xr / (R * Math.cos(centerLat * Math.PI / 180));

            const newLat = centerLat + dLat_point * (180 / Math.PI);
            const newLng = centerLng + dLng_point * (180 / Math.PI);

            coords.push(new kakao.maps.LatLng(newLat, newLng));
        }

        currentPolygon = new kakao.maps.Polygon({
            path: coords,
            strokeWeight: 2,
            strokeColor: '#FF4E4E',
            strokeOpacity: 0.9,
            strokeStyle: 'solid',
            fillColor: '#FF8888',
            fillOpacity: 0.5
        });

        currentPolygon.setMap(map);
    }

    function addDirectionArrowOnMap(map, lat, lng, direction) {
        if (currentArrow) currentArrow.setMap(null);
        if (lat === -999 || lng === -999) return;

        const angleMap = { N: 0, NE: 45, E: 90, SE: 135, S: 180, SW: 225, W: 270, NW: 315 };
        const angle = angleMap[direction] || 0;

        const content = `
            <div style="
                transform: rotate(${angle}deg);
                width: 30px; height: 30px;
                background: url('/WildFire/img/arrow-icon.png') no-repeat center;
                background-size: contain;">
            </div>
        `;

        currentArrow = new kakao.maps.CustomOverlay({
            content: content,
            position: new kakao.maps.LatLng(lat, lng),
            yAnchor: 0.5
        });

        currentArrow.setMap(map);
    }

    function addPredictionLabelOnMap(map, lat, lng, damageArea, direction, totalDistance, speedValue, speedCategory) {
        if (currentLabel) currentLabel.setMap(null);
        if (lat === -999 || lng === -999) return;

        const directionAngles = {
            N: 0, NE: 45, E: 90, SE: 135,
            S: 180, SW: 225, W: 270, NW: 315
        };
        const angleDeg = directionAngles[direction] || 0;
        const angleRad = angleDeg * Math.PI / 180;

        const area_m2 = damageArea * 10000;
        const aspectRatio = 0.6;
        const b = Math.sqrt(area_m2 / (Math.PI * aspectRatio));
        const a = b / aspectRatio;

        const x = a * Math.cos(angleRad);
        const y = b * Math.sin(angleRad);

        const R = 6371000;
        const dLat = y / R;
        const dLng = x / (R * Math.cos(lat * Math.PI / 180));

        const edgeLat = lat + dLat * (180 / Math.PI);
        const edgeLng = lng + dLng * (180 / Math.PI);

        const content = `
            <div style="
                background: rgba(255,255,255,0.95);
                border: 1px solid #999;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                color: #333;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            ">
                피해 면적: <strong>${damageArea.toFixed(2)} ha</strong><br/>
                확산 방향: <strong>${direction}</strong><br/>
                확산 거리: <strong>${totalDistance.toFixed(2)} m</strong><br/>
                확산 속도: <strong>${speedValue.toFixed(2)} m/h</strong> (${speedCategory})
            </div>
        `;

        currentLabel = new kakao.maps.CustomOverlay({
            content: content,
            position: new kakao.maps.LatLng(edgeLat, edgeLng),
            yAnchor: 1.2
        });

        currentLabel.setMap(map);
    }
});
