<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>SEED - Smart Ecological Evaluation & Diagnostics</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="/WildFire/img/logo_leaf.png" type="image/png">
  <link rel="stylesheet" href="/WildFire/css/recoveryMain.css">

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <link href="https://cdn.jsdelivr.net/npm/tom-select@2.3.1/dist/css/tom-select.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/tom-select@2.3.1/dist/js/tom-select.complete.min.js"></script>

  <script src="//dapi.kakao.com/v2/maps/sdk.js?appkey=946126d34d2237e3e7e7360f97674a47&autoload=false"></script>
</head>
<body>
<div class="container">

  <jsp:include page="header.jsp" />

  <section id="region-select-wrapper" class="region-select">
    <label for="city">지역 선택:</label>
    <select id="city" class="ts-select">
      <option value="">시/군 선택</option>
      <option value="강릉시">강릉시</option>
      <option value="고성군">고성군</option>
      <option value="동해시">동해시</option>
      <option value="삼척시">삼척시</option>
      <option value="속초시">속초시</option>
      <option value="양구군">양구군</option>
      <option value="양양군">양양군</option>
      <option value="영월군">영월군</option>
      <option value="원주시">원주시</option>
      <option value="인제군">인제군</option>
      <option value="정선군">정선군</option>
      <option value="철원군">철원군</option>
      <option value="춘천시">춘천시</option>
      <option value="태백시">태백시</option>
      <option value="평창군">평창군</option>
      <option value="홍천군">홍천군</option>
      <option value="화천군">화천군</option>
      <option value="횡성군">횡성군</option>
    </select>
    <button id="confirmBtn" class="filter-button">선택 완료</button>
  </section>

  <main class="main-content-horizontal">

    <section class="map-section">
      <div id="map" style="width: 100%; height: 550px;"></div>
    </section>

    <div class="bar-chart-box">
      <h4>변수 영향력 (%)</h4>
      <canvas id="barChart"></canvas>
    </div>

    <div class="top3-box">
      <h4>NDVI 영향 TOP 3 변수</h4>
      <div id="top3-variables"class="top3-variable-container"></div>

      <div class="forest-slide-wrapper">
        <h4 id="forest-title">강원도 산림 현황</h4>
        <div class="forest-card-slider">
          <button id="forest-prev" class="forest-arrow">&#8249;</button>
          <div class="forest-slide-list"></div>
          <button id="forest-next" class="forest-arrow">&#8250;</button>
        </div>
      </div>
    </div>
  </main>

  <section class="bottom-section">
    <div class="ndvi-graph-box">
      <h4 id="ndvi-title">NDVI 변화율 (2001년~2022년까지)</h4>
      <canvas id="ndviChart"></canvas>
    </div>
  </section>
  <%
  boolean isLoggedIn = (session.getAttribute("user") != null);
%>
<section class="download-section">
  <h4>결과 다운로드</h4>
  <div class="download-buttons">
    <a href="<%= isLoggedIn ? "/WildFire/DownloadCSVServlet" : "#" %>" 
       class="btn-download download-link"
       data-type="csv">
      <img src="/WildFire/img/icon_csv.png" class="btn-icon" alt="CSV 아이콘">
      CSV 다운로드
    </a>
    <a href="<%= isLoggedIn ? "/WildFire/DownloadPDFServlet" : "#" %>" 
       class="btn-download download-link"
       data-type="pdf">
      <img src="/WildFire/img/icon_pdf.png" class="btn-icon" alt="PDF 아이콘">
      PDF 다운로드
    </a>
    </div>
</section>

<jsp:include page="footer.jsp" />
</div>
<script>
  window.isLoggedIn = <%= (session.getAttribute("user") != null) %>;
</script>
<script src="/WildFire/js/map.js"></script>
<script src="/WildFire/js/select.js" defer></script>
<script src="/WildFire/js/recoveryMain.js" defer></script>
</body>
</html>