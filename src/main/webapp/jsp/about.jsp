<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>About - SEED</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
	<link rel="icon" href="/WildFire/img/logo_leaf.png" type="image/png">
  <!-- CSS -->
  <link rel="stylesheet" href="/WildFire/css/about.css">
</head>
<body>
<div class="container">
<!-- ✅ 헤더 -->
<jsp:include page="header.jsp" />

<!-- ✅ About Us 섹션 -->
<main class="main-content">
  <section class="feature-section">
    <h1>SEED는 이런 일을 해요</h1>
    <div class="feature-grid">
      <div class="feature-card">
        <img src="/WildFire/img/fire_icon.png" alt="산불 알림 아이콘">
        <h3>산불 현황 알림</h3>
        <p>실시간 산불 발생 위치와<br>위험 지역 정보를 알려드립니다.</p>
      </div>
      <div class="feature-card">
        <img src="/WildFire/img/icon_fire.png" alt="산불 예측 아이콘">
        <h3>산불 확산 경로 예측</h3>
        <p>기상·지형 데이터를 바탕으로 <br>사용자가 선택한 위치에서의 <span class="no-break">산불 확산 경로와 속도를 예측합니다.</span></p>
      </div>
      <div class="feature-card">
        <img src="/WildFire/img/icon_forest.png" alt="복원력 분석 아이콘">
        <h3>산림 복원력 예측</h3>
        <p>NDVI 시계열 분석을 통해<br>산림의 회복 경향을 시각적으로 <span class="no-break">예측하여 보여줍니다.</span></p>
      </div>
      <div class="feature-card">
        <img src="/WildFire/img/icon_ai.png" alt="AI 조언 아이콘">
        <h3>AI 기반 복원 조언</h3>
        <p>회복이 느린 지역에 대해<br>AI가 최적의 조치를 제안합니다.</p>
      </div>
    </div>
  </section>
</main>

<!-- ✅ 푸터 -->
<jsp:include page="footer.jsp" />
</div>
</body>
</html>