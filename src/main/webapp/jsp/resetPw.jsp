<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>비밀번호 재설정 - SEED</title>
  <link rel="stylesheet" href="/WildFire/css/main.css">
  <link rel="stylesheet" href="/WildFire/css/login.css">
  <link rel="icon" href="/WildFire/img/logo_leaf.png" type="image/png">
</head>
<body>
<div class="container">
  <div class="login-wrapper">
    <div class="login-box">
      <a href="main.jsp" class="logo-link">
        <img src="/WildFire/img/logo.png" alt="SEED 로고" class="login-logo">
      </a>

      <h2>비밀번호 재설정</h2>
      <form action="/WildFire/ResetPwServlet" method="post">
        <input type="text" name="username" placeholder="아이디" required>
        <input type="email" name="email" placeholder="이메일" required>
        <input type="password" name="newPassword" placeholder="새 비밀번호" required>
        <input type="password" name="confirmPassword" placeholder="비밀번호 확인" required>
        <button type="submit">변경하기</button>
      </form>

      <div class="switch">
        <a href="login.jsp">로그인으로 돌아가기</a>
      </div>
    </div>
  </div>
</div>
</body>
</html>