<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>SEED 로그인</title>
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

        <form action="/WildFire/LoginServlet" method="post">
          <input type="text" name="username" placeholder="아이디" required>
          <input type="password" name="password" placeholder="비밀번호" required>
          <input type="hidden" name="redirect" value="<%= request.getParameter("redirect") != null ? request.getParameter("redirect") : "" %>">
          <button type="submit">로그인</button>
          <div class="login-options">
            <label><input type="checkbox" name="remember"> 로그인 상태 유지</label>
          </div>
        </form>

        <div class="switch">
          <a href="findId.jsp">아이디 찾기</a> &nbsp;|&nbsp;
          <a href="resetPw.jsp">비밀번호 찾기</a> &nbsp;|&nbsp;
          <a href="signup.jsp">회원가입</a>
        </div>
      </div>
    </div>
  </div>
</body>
</html>