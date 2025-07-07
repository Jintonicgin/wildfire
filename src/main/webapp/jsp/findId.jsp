<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>아이디 찾기 - SEED</title>
  <link rel="stylesheet" href="/WildFire/css/main.css">
  <link rel="stylesheet" href="/WildFire/css/login.css">
</head>
<body>
<div class="container">
  <div class="login-wrapper">
    <div class="login-box">
      <a href="main.jsp" class="logo-link">
        <img src="/WildFire/img/logo.png" alt="SEED 로고" class="login-logo">
      </a>

      <h2>아이디 찾기</h2>
      <form action="/WildFire/FindIdServlet" method="post">
        <input type="text" name="name" placeholder="이름" required>
        <input type="email" name="email" placeholder="이메일" required>
        <button type="submit">아이디 찾기</button>
      </form>

      <div class="switch">
        <a href="login.jsp">로그인으로 돌아가기</a>
      </div>
    </div>
  </div>
</div>
</body>
</html>