<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%
  if (session.getAttribute("user") == null) {
    Cookie[] cookies = request.getCookies();
    if (cookies != null) {
      for (Cookie c : cookies) {
        if (c.getName().equals("user")) {
          session.setAttribute("user", c.getValue());
          break;
        }
      }
    }
  }

  String user = (String) session.getAttribute("user");
%>
<header class="top-bar desktop-header">
  <div class="logo-area">
    <a href="main.jsp">
      <img src="/WildFire/img/logo.png" alt="SEED 로고" class="logo-horizontal">
    </a>
  </div>
  
  <nav class="main-nav">
    <a href="fire.jsp">산불현황</a>
    <a href="recoveryMain.jsp">복원예측</a>
    <a href="#">AI 복원 조언</a>
  </nav>
  <div class="auth-links">
    <a href="about.jsp">ABOUT US</a>
   <% if (user == null) { %>
  <a href="login.jsp">로그인 / 회원가입</a>
<% } else { %>
  <a href="/WildFire/LogoutServlet">로그아웃</a>
<% } %>
  </div>
</header>

<div class="mobile-header">
  <div class="mobile-header-top">
    <a href="main.jsp">
      <img src="/WildFire/img/logo.png" class="mobile-logo" alt="로고">
    </a>
  </div>
  <div class="mobile-menu-card">
    <a href="fire.jsp">산불현황</a>
    <a href="recoveryMain.jsp">복원예측</a>
    <a href="#">AI 복원 조언</a>
    <a href="about.jsp">ABOUT US</a>
    <% if (user == null) { %>
  <a href="login.jsp">로그인 / 회원가입</a>
<% } else { %>
  <a href="/WildFire/LogoutServlet">로그아웃</a>
<% } %>
  </div>
</div>

<style>
.header-wrapper {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

.top-bar {
  margin-top: 30px;
  display: flex;
  align-items: center;
  position: relative;
  padding: 16px 32px;
  background-color: #a8d8b6;
  border-radius: 12px;
}

.mobile-header {
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 20px;
  gap: 12px;
  background-color: #a8d8b6;
  border-radius: 20px;
}

.logo-horizontal,
.mobile-logo {
  height: 52px;
}

.main-nav {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 60px;
}

.main-nav a {
  margin: 0 20px;
  text-decoration: none;
  color: #264638;
  font-weight: 600;
  font-size: 25px;
}
.main-nav a:hover {
  color: #357658;
}

.auth-links {
  margin-left: auto;
  display: flex;
  gap: 14px; 
  align-items: center;
}

.auth-links a {
  text-decoration: none;
  color: #264638;
  font-size: 16px; 
  margin: 0 4px; 
}

.auth-links a:hover {
  color: #000;
}

.mobile-header {
  display: none;
  flex-direction: column;
  align-items: flex-start;
  padding: 16px 24px;
  gap: 12px;
  background-color: #a8d8b6;
  border-radius: 20px;
  margin-top: 0;
}

.mobile-header-top {
  display: flex;
  align-items: center;
  gap: 12px;
}

.mobile-menu-card {
  background-color: #eaf6ee;
  border-radius: 20px;
  padding: 24px 36px;
  width: 89%;
  max-width: 720px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  display: flex;
  flex-direction: column;
  gap: 17px;
  align-items: flex-start;
}

.mobile-menu-card a {
  text-decoration: none;
  font-weight: 700;
  font-size: 18px;
  color: #264638;
  transition: all 0.2s;
}
.mobile-menu-card a:hover {
  color: #000000;
}

@media screen and (max-width: 768px) {
  .desktop-header {
    display: none;
  }

  .mobile-header {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 16px 24px;
    gap: 10px;
    margin-top: 24px;
  }
}

@media screen and (min-width: 769px) {
  .mobile-header {
    display: none;
  }
}
</style>