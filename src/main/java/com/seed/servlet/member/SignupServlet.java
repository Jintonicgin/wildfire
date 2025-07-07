package com.seed.servlet.member;

import com.seed.dao.MemberDAO;
import com.seed.model.Member;
import com.seed.util.SHA256;
import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;

@WebServlet("/SignupServlet")
public class SignupServlet extends HttpServlet {
  private static final long serialVersionUID = 1L;

  protected void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    request.setCharacterEncoding("UTF-8");
    response.setContentType("text/html; charset=UTF-8");

    String username = request.getParameter("username");
    String password = request.getParameter("password");
    String confirm = request.getParameter("confirmPassword");
    String email = request.getParameter("email");
    String name = request.getParameter("name");

    if (username == null || password == null || confirm == null || email == null || name == null ||
        username.isEmpty() || password.isEmpty() || confirm.isEmpty() || email.isEmpty() || name.isEmpty()) {
      sendAlert(response, "모든 항목을 입력해주세요.");
      return;
    }

    if (!password.equals(confirm)) {
      sendAlert(response, "비밀번호가 일치하지 않습니다.");
      return;
    }

    MemberDAO dao = new MemberDAO();
    if (dao.isUsernameTaken(username)) {
      sendAlert(response, "이미 사용 중인 아이디입니다.");
      return;
    }

    if (dao.isEmailTaken(email)) {
      sendAlert(response, "이미 가입된 이메일입니다.");
      return;
    }

    String encryptedPw = SHA256.encrypt(password);
    Member member = new Member(username, encryptedPw, email, name);

    boolean success = dao.insertMember(member);
    if (success) {
      response.sendRedirect("/WildFire/jsp/signupSuccess.jsp"); // context root 확인
    } else {
      sendAlert(response, "회원가입에 실패했습니다. 다시 시도해주세요.");
    }
  }

  private void sendAlert(HttpServletResponse response, String message) throws IOException {
    response.getWriter().println("<script>alert('" + message + "'); history.back();</script>");
  }
}