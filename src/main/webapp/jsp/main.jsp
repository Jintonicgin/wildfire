<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>SEED - Smart Ecological Evaluation & Diagnostics</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <link rel="stylesheet" href="/WildFire/css/main.css">
</head>
<body>

<div class="intro-screen">
  <img src="/WildFire/img/logo_leaf.png" alt="로고" class="intro-logo">
  <div class="intro-text-wrapper">
    <div class="intro-text" id="type-text"></div>
    <div class="intro-subtitle" id="type-sub" style="visibility: hidden;"></div>
  </div>
  <button class="intro-start-btn" id="enterBtn" style="display: none;">시작하기</button>
</div>

<div class="container">
<jsp:include page="header.jsp" />

<div class="main-content">

<section class="forest-banner"></section>

<section class="hero-section">
  <div class="hero-overlay">
    <h1><span class="no-break">데이터로 자연의 회복을 예측하다</span></h1>
    <p>산불 이후, 우리가 할 수 있는 일</p>
  </div>
</section>

<section class="features">
  <h2 class="section-title">SEED가 하는 일</h2>
  <div class="feature-list">
    <div class="feature-item scroll-fade">
      <div class="emoji">🔥</div>
      <h3>산불 현황 알림</h3>
      <p>실시간 위치 및 위험도 제공</p>
    </div>
    <div class="feature-item scroll-fade">
      <div class="emoji">🌳</div>
      <h3>복원력 분석</h3>
      <p>NDVI 기반 회복력 추적</p>
    </div>
    <div class="feature-item scroll-fade">
      <div class="emoji">🤖</div>
      <h3>AI 복원 조언</h3>
      <p>빠른 대응을 위한 AI 전략</p>
    </div>
  </div>
</section>

<section class="scenario">
  <h2 class="section-title">사용자 시나리오 미리보기</h2>
  <div class="scenario-box scroll-fade">
    <p>“2024년 5월 — 삼척시 인근 산불 발생”</p>
    <p>실시간 위험 지역, 발생 위치 지도 시각화</p>
    <a href="fire.jsp" class="scenario-button">🔥 산불 현황 보기 →</a>
  </div>

  <div class="scenario-box scroll-fade">
    <p>“속초시 — 회복률 <strong>83%</strong>”</p>
    <p>NDVI 선 그래프 및 영향 요인 시각화</p>
    <a href="recoveryMain.jsp" class="scenario-button">📊 지금 내 지역 복원력 보기 →</a>
  </div>
</section>

<section class="slogan scroll-fade">
  <blockquote>
    숲은 다시 살아납니다.<br />
    <strong>데이터와 함께라면.</strong>
  </blockquote>
</section>

<jsp:include page="footer.jsp" />
</div>
</div>
<!-- JS -->
<script src="/WildFire/js/main.js"></script>
</body>
</html>