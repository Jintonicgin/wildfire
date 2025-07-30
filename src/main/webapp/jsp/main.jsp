<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>SEED - Smart Ecological Evaluation & Diagnostics</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <!-- Bootstrap JS -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  <!-- Custom CSS -->
  <link rel="stylesheet" href="/WildFire/css/main.css">
  <link rel="icon" href="/WildFire/img/logo_leaf.png" type="image/png">
</head>
<body>

<!-- Main Container -->
<div class="container">
  <jsp:include page="header.jsp" />

  <div class="main-content">

    <!-- Hero Section (Carousel) -->
    <section class="hero-section carousel slide" id="heroCarousel" data-bs-ride="carousel" data-bs-interval="5000" data-bs-pause="hover">
      <div class="carousel-inner">

        <!-- Slide 1: 산불 현황 -->
        <div class="carousel-item active">
		  <img src="/WildFire/img/forest4.png" class="d-block w-100 hero-image" alt="산불현황">
		  <div class="carousel-caption hero-caption">
		    <h1>산불 현황 알림 서비스</h1>
		    <p>실시간 위험 지역과 산불 발생 현황을 알려드립니다.</p>
		    <a href="fire.jsp" class="hero-btn">🔥 산불 현황 보러가기</a>
		  </div>
		</div>
       

        <!-- Slide 2: 산불 예측 -->
        <div class="carousel-item">
		  <img src="/WildFire/img/forest2.png" class="d-block w-100 hero-image" alt="산불예측">
		  <div class="carousel-caption hero-caption">
		    <h1>데이터로 산불을 예측하다</h1>
		    <p>강원도를 중심으로 <span class="no-break">산불의 확산 경로와 속도, 피해면적을 예측합니다.</span></p>
		    <a href="prediction.jsp" class="hero-btn">🧭 산불 예측하러가기</a>
		  </div>
		</div>

        <!-- Slide 3: 복원 예측 -->
        <div class="carousel-item">
		  <img src="/WildFire/img/forest.png" class="d-block w-100 hero-image" alt="복원예측">
		  <div class="carousel-caption hero-caption">
		    <h1><span class="no-break">데이터로 자연의 회복을 예측하다</span></h1>
		    <p>산불 이후, 우리가 할 수 있는 일</p>
		    <a href="recoveryMain.jsp" class="hero-btn">🌳 산림 복원 예측하러가기</a>
		  </div>
		</div>

        <!-- Slide 4: AI 복원 조언 -->
        <div class="carousel-item">
		  <img src="/WildFire/img/forest3.png" class="d-block w-100 hero-image" alt="복원조언">
		  <div class="carousel-caption hero-caption">
		    <h1>AI가 제안하는 복원 전략</h1>
		    <p>데이터 기반의 복원 우선 지역과 전략 추천</p>
		    <a href="aiAdvice.jsp" class="hero-btn">🤖 AI 복원 조언 받으러가기</a>
		  </div>
		</div>

      <!-- Carousel Buttons -->
      <button class="carousel-control-prev" type="button" data-bs-target="#heroCarousel" data-bs-slide="prev">
        <span class="carousel-control-prev-icon" aria-hidden="true"></span>
      </button>
      <button class="carousel-control-next" type="button" data-bs-target="#heroCarousel" data-bs-slide="next">
        <span class="carousel-control-next-icon" aria-hidden="true"></span>
      </button>
    </section>

    <!-- 기능 소개 -->
    <section class="features">
      <h2 class="section-title">SEED가 하는 일</h2>
      <div class="feature-list">
        <div class="feature-item scroll-fade">
          <div class="emoji">🔥</div>
          <h3>산불 현황 알림</h3>
          <p>실시간 위치 및 위험도 제공</p>
        </div>
        <div class="feature-item scroll-fade">
          <div class="emoji">🧭️</div>
          <h3>산불 예측</h3>
          <p>산불 확산 경로 시뮬레이션</p>
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

    <!-- 시나리오 미리보기 -->
    <section class="scenario">
      <h2 class="section-title">사용자 시나리오 미리보기</h2>
      <div class="scenario-box scroll-fade">
        <p>“2024년 5월 — 삼척시 인근 산불 발생”</p>
        <p>실시간 위험 지역, 발생 위치 지도 시각화</p>
        <a href="fire.jsp" class="scenario-button">🔥 산불 현황 보기 →</a>
      </div>
      <div class="scenario-box scroll-fade">
		<p>“산불 예측, 이제는 데이터가 말해줍니다”</p>
		<p>위치 기반 기상·지형 정보로 확산 방향과 속도까지 예측</p>
		<a href="prediction.jsp" class="scenario-button">🧭 산불 확산 예측 보기 →</a>
	 </div>
      <div class="scenario-box scroll-fade">
        <p>“속초시 — 회복률 <strong>83%</strong>”</p>
        <p>NDVI 선 그래프 및 영향 요인 시각화</p>
        <a href="recoveryMain.jsp" class="scenario-button">📊 지금 내 지역 복원력 보기 →</a>
      </div>
    </section>

    <!-- 슬로건 -->
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