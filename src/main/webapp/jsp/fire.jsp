<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>산불 현황 - SEED</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="icon" href="/WildFire/img/logo_leaf.png" type="image/png">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<link href="https://cdn.jsdelivr.net/npm/tom-select@2.3.1/dist/css/tom-select.css" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/tom-select@2.3.1/dist/js/tom-select.complete.min.js"></script>
  <link rel="stylesheet" href="/WildFire/css/fire.css">

<script src="//dapi.kakao.com/v2/maps/sdk.js?appkey=946126d34d2237e3e7e7360f97674a47&autoload=false"></script>
</head>
<body>
<div class="container">
  <jsp:include page="header.jsp" />

<section class="filter-wrapper">
  <label for="period" class="filter-label">조회 기간:</label>
  <select id="period" class="ts-select">
    <option value="">기간 선택</option>
    <option value="today">오늘</option>
    <option value="3d">최근 3일</option>
    <option value="7d">최근 1주</option>
  </select>
  <button id="periodConfirmBtn" class="filter-button">선택 완료</button>
</section>


  <section class="fire-content-wrapper">
    <!-- 지도 -->
<div class="fire-map-box">
  <div id="fireMap" style="width: 100%; height: 500px;"></div>
</div>

    <div class="fire-info-box fire-info-wrapper">
      <div class="fire-list">
        <h3>산불 발생 목록</h3>
        <ul>
          <li>2025.06.30 - 강릉시 왕산면</li>
          <li>2025.06.29 - 평창군 봉평면</li>
          <li>2025.06.28 - 정선군 고한읍</li>
        </ul>
      </div>
      <div class="fire-alert">
        <h3>위험 지역 알림</h3>
        <p>강풍주의보 발효 중: 강릉시, 속초시</p>
        <p>건조주의보 발효 중: 양양군</p>
      </div>
      <div class="fire-highlight">
        <h3>최근 최대 산불 발생 지역</h3>
        <p>📍 2025.04.12 - 삼척시 근덕면</p>
        <p>피해 면적 약 312ha / 원인: 입산자 실화</p>
      </div>
    </div>
  </section>

<section class="fire-chart-section">
  <h3>최근 5년간 산불 발생 추이</h3>
  <canvas id="yearlyFireChart"></canvas>
</section>

  <jsp:include page="footer.jsp" />
</div>
<script src="/WildFire/js/firemap.js" defer></script>
<script src="/WildFire/js/fire.js"></script>
</body>
</html>