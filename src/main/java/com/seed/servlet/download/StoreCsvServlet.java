package com.seed.servlet.download;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;

@WebServlet("/StoreCsvServlet")
public class StoreCsvServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        request.setCharacterEncoding("UTF-8");

        BufferedReader reader = request.getReader();
        StringBuilder jsonBuilder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            jsonBuilder.append(line);
        }

        String raw = jsonBuilder.toString()
                .replace("[[", "")
                .replace("]]", "")
                .replace("],[", "#")
                .replace("\"", "");

        String[] rows = raw.split("#");

        List<String[]> parsedList = new ArrayList<>();
        for (String row : rows) {
            parsedList.add(row.split(","));
        }

        HttpSession session = request.getSession();
        session.setAttribute("csvData", parsedList);

        response.setStatus(200);
    }
}