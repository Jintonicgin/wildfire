package com.seed.servlet.download;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;
import java.io.*;
import java.net.URLEncoder;
import java.util.List;

@WebServlet("/DownloadCSVServlet")
public class DownloadCSVServlet extends HttpServlet {
  private static final long serialVersionUID = 1L;

  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    HttpSession session = request.getSession(false);
    String user = (session != null) ? (String) session.getAttribute("user") : null;

    if (user == null) {
      response.sendRedirect("login.jsp");
      return;
    }

    String region = request.getParameter("region");
    String filename = (region != null && !region.isEmpty()) ? region + "_forest_recovery.csv" : "forest_recovery.csv";
    List<String[]> csvData = (List<String[]>) session.getAttribute("csvData");

    if (csvData == null || csvData.isEmpty()) {
    	  System.out.println("CSV 데이터 없음. csvData = " + csvData);
    	  response.setContentType("text/plain; charset=UTF-8");
    	  response.getWriter().println("다운로드할 CSV 데이터가 없습니다.");
    	  return;
    	}

    String encodedFilename = URLEncoder.encode(filename, "UTF-8").replaceAll("\\+", "%20");

    response.setCharacterEncoding("UTF-8");
    response.setContentType("text/csv; charset=UTF-8");
    response.setHeader("Content-Disposition", "attachment; filename=\"" + encodedFilename + "\"");

    try (OutputStream out = response.getOutputStream();
         OutputStreamWriter osw = new OutputStreamWriter(out, "UTF-8");
         PrintWriter writer = new PrintWriter(osw)) {

      // BOM 출력 (Excel 호환용)
      writer.write('\uFEFF');

      for (String[] row : csvData) {
        writer.println(String.join(",", row));
      }
    }
  }
}