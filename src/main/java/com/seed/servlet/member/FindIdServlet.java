package com.seed.servlet.member;

import com.seed.dao.MemberDAO;
import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet({"/FindIdServlet"})
public class FindIdServlet extends HttpServlet {
  private static final long serialVersionUID = 1L;

  protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    request.setCharacterEncoding("UTF-8");
    response.setContentType("text/html; charset=UTF-8");

    String name = request.getParameter("name");
    String email = request.getParameter("email");

    MemberDAO dao = new MemberDAO();
    String foundId = dao.findUsernameByNameAndEmail(name, email);

    if (foundId != null) {
      response.getWriter().println("<script>alert('찾은 아이디는 " + foundId + " 입니다.'); history.back();</script>");
    } else {
      response.getWriter().println("<script>alert('일치하는 정보가 없습니다.'); history.back();</script>");
    }
  }
}