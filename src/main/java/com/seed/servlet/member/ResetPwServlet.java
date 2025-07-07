package com.seed.servlet.member;

import com.seed.dao.MemberDAO;
import com.seed.util.SHA256;
import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet({"/ResetPwServlet"})
public class ResetPwServlet extends HttpServlet {
  private static final long serialVersionUID = 1L;

  protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    request.setCharacterEncoding("UTF-8");
    response.setContentType("text/html; charset=UTF-8");

    String username = request.getParameter("username");
    String email = request.getParameter("email");
    String newPw = request.getParameter("newPassword");
    String confirmPw = request.getParameter("confirmPassword");

    if (!newPw.equals(confirmPw)) {
      response.getWriter().println("<script>alert('비밀번호가 일치하지 않습니다.'); history.back();</script>");
      return;
    }

    MemberDAO dao = new MemberDAO();
    boolean valid = dao.checkUserByIdAndEmail(username, email);

    if (!valid) {
      response.getWriter().println("<script>alert('아이디와 이메일이 일치하지 않습니다.'); history.back();</script>");
      return;
    }

    String encryptedPw = SHA256.encrypt(newPw);
    boolean updated = dao.updatePassword(username, encryptedPw);

    if (updated) {
      response.getWriter().println("<script>alert('비밀번호가 성공적으로 변경되었습니다.'); location.href='/WildFire/jsp/login.jsp';</script>");
    } else {
      response.getWriter().println("<script>alert('비밀번호 변경에 실패했습니다.'); history.back();</script>");
    }
  }
}