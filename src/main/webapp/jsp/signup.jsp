<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>회원가입 - SEED</title>
  <link rel="stylesheet" href="/WildFire/css/main.css">
  <link rel="stylesheet" href="/WildFire/css/login.css">
</head>
<body>
  <div class="container">
    <div class="login-wrapper">
      <div class="login-box">
        <!-- 로고 -->
        <a href="main.jsp" class="logo-link">
          <img src="/WildFire/img/logo.png" alt="SEED 로고" class="login-logo">
        </a>

        <h2>회원가입</h2>
        <form id="signupForm" action="/WildFire/SignupServlet" method="post">
        <input type="text" name="name" placeholder="이름" required>
          <input type="email" name="email" placeholder="이메일" required>
          <input type="text" name="username" placeholder="아이디" required>
          <input type="password" name="password" placeholder="비밀번호" required>
          <input type="password" name="confirmPassword" placeholder="비밀번호 확인" required>

          <button type="submit">회원가입</button>
        </form>

        <div class="switch">
          이미 계정이 있으신가요?
          <a href="login.jsp">로그인</a>
        </div>
      </div>
    </div>
  </div>
  <script src="/WildFire/js/signup.js"></script>
</body>
</html>