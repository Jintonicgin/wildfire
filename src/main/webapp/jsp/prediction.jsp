<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%
    boolean isLoggedIn = (session.getAttribute("user") != null);
%>

<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>산불 예측 - SEED</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	<link rel="icon" href="/WildFire/img/logo_leaf.png" type="image/png">
    <!-- CSS -->
    <link rel="stylesheet" href="/WildFire/css/main.css">
    <link rel="stylesheet" href="/WildFire/css/prediction.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tom-select@2.3.1/dist/css/tom-select.css">

    <!-- JS -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="//dapi.kakao.com/v2/maps/sdk.js?appkey=946126d34d2237e3e7e7360f97674a47&autoload=false"></script>
</head>
<body>
<div class="container">
    <jsp:include page="header.jsp" />

    <div class="prediction-container">

        <!-- 지도 -->
        <div id="predictionMap" style="width: 100%; height: 400px; margin-top: 10px;"></div>

        <!-- 선택 정보 및 예측 실행 -->
        <form id="predictionForm" class="predict-row">
            <div class="coord-group">
                <label>위도:
                    <input type="text" id="latitude" name="latitude" readonly>
                </label>
                <label>경도:
                    <input type="text" id="longitude" name="longitude" readonly>
                </label>
            </div>

            <div class="time-group">
                <label>예측 시간:
                    <select id="forecastHour" name="forecastHour" placeholder="예측 시간 선택" autocomplete="off">
                        <option value="">예측 시간 선택</option>
                        <option value="3">3시간 후</option>
                        <option value="6">6시간 후</option>
                        <option value="9">9시간 후</option>
                        <option value="12">12시간 후</option>
                    </select>
                </label>
            </div>

            <button type="submit" id="predictBtn">🔥 예측 실행</button>
        </form>

        <!-- 로딩 오버레이 -->
        <div id="loadingOverlay" class="hidden">
            <div class="spinner"></div>
            <p>예측 중입니다...</p>
        </div>

        <!-- 예측 결과 -->
        <div class="prediction-result">
            <h3>🧾 예측 결과</h3>
            <p>피해 면적: <span id="predictedDamageArea">-</span></p>
            <p>확산 방향 및 속도:</p>
            <div id="predictedSpreadDirection" class="direction-box"></div>
        </div>
    </div>

    <jsp:include page="footer.jsp" />
</div>

<!-- Scripts -->
<script src="https://cdn.jsdelivr.net/npm/tom-select@2.3.1/dist/js/tom-select.complete.min.js"></script>
<script src="/WildFire/js/prediction.js"></script>

<!-- ✅ 로그인 상태를 JS 전역 변수로 전달 -->
<script>
    window.isLoggedIn = <%= isLoggedIn %>;
</script>

</body>
</html>