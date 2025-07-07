package com.seed.servlet.download;

import java.io.*;
import java.util.*;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;

@WebServlet("/SetCsvDataServlet")
public class SetCsvDataServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        request.setCharacterEncoding("UTF-8");
        response.setContentType("text/plain; charset=UTF-8");

        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = request.getReader()) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }

        // 예시: [[\"a\",\"b\"],[\"c\",\"d\"]] 와 같은 문자열이라고 가정
        String jsonString = sb.toString();

        List<String[]> csvData = new ArrayList<>();
        try {
            jsonString = jsonString.trim();
            if (jsonString.startsWith("[") && jsonString.endsWith("]")) {
                // 중첩 배열 처리
                jsonString = jsonString.substring(1, jsonString.length() - 1); // 바깥쪽 []
                String[] rows = jsonString.split("\\],\\[");

                for (String row : rows) {
                    row = row.replaceAll("^\\[|\\]$", ""); // 남아있는 괄호 제거
                    row = row.replaceAll("\"", ""); // 따옴표 제거
                    String[] cols = row.split(",");
                    csvData.add(cols);
                }

                if (!csvData.isEmpty()) {
                    HttpSession session = request.getSession();
                    session.setAttribute("csvData", csvData);
                    response.setStatus(HttpServletResponse.SC_OK);
                    response.getWriter().write("csvData 저장 성공");
                } else {
                    response.setStatus(HttpServletResponse.SC_NO_CONTENT);
                    response.getWriter().write("csvData가 비어 있습니다");
                }
            } else {
                response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                response.getWriter().write("JSON 배열 형식이 아님");
            }

        } catch (Exception e) {
            e.printStackTrace();
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            response.getWriter().write("서버 오류: " + e.getMessage());
        }
    }
}