package com.seed.servlet.member;

import com.seed.dao.MemberDAO;
import com.seed.util.SHA256;
import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

@WebServlet({"/LoginServlet"})
public class LoginServlet extends HttpServlet {
  private static final long serialVersionUID = 1L;

  protected void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    request.setCharacterEncoding("UTF-8");
    response.setContentType("text/html; charset=UTF-8");

    String username = request.getParameter("username");
    String password = request.getParameter("password");
    String remember = request.getParameter("remember");

    if (username == null || password == null || username.isEmpty() || password.isEmpty()) {
      response.getWriter().println("<script>alert('아이디와 비밀번호를 모두 입력해주세요.'); history.back();</script>");
      return;
    }

    String encryptedPw = SHA256.encrypt(password);
    MemberDAO dao = new MemberDAO();

    if (dao.validateLogin(username, encryptedPw)) {
      HttpSession session = request.getSession();
      session.setAttribute("user", username);

      if ("on".equals(remember)) {
        Cookie cookie = new Cookie("user", username);
        cookie.setMaxAge(604800); // 7일
        cookie.setPath("/");
        response.addCookie(cookie);
      }

      response.sendRedirect("/WildFire/jsp/main.jsp");

    } else {
      response.getWriter().println("<script>alert('아이디 또는 비밀번호가 일치하지 않습니다.'); history.back();</script>");
    }
  }
}