package com.seed.servlet;

import org.json.JSONObject;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.*;

@WebServlet("/PredictionServlet")
public class PredictionServlet extends HttpServlet {

    private static final long serialVersionUID = 1L;

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        request.setCharacterEncoding("UTF-8");
        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");

        // ✅ CORS 허용
        response.setHeader("Access-Control-Allow-Origin", "*");
        response.setHeader("Access-Control-Allow-Methods", "POST");
        response.setHeader("Access-Control-Allow-Headers", "Content-Type");

        PrintWriter out = response.getWriter();

        try {
            // ✅ 1. 요청 JSON 파싱
            StringBuilder sb = new StringBuilder();
            BufferedReader reader = request.getReader();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }

            JSONObject jsonRequest = new JSONObject(sb.toString());

            // ✅ 2. Python 실행 및 결과 수신
            String resultJson = runPythonPrediction(jsonRequest);

            if (resultJson == null || resultJson.isEmpty()) {
                response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
                JSONObject error = new JSONObject();
                error.put("error", "Python 예측 결과가 비어 있습니다.");
                out.print(error.toString());
                return;
            }

            // ✅ 3. 결과 응답
            out.print(resultJson);

        } catch (Exception e) {
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            JSONObject error = new JSONObject();
            error.put("error", "서버 처리 중 예외 발생: " + e.toString());
            out.print(error.toString());
            e.printStackTrace();
        } finally {
            out.flush();
            out.close();
        }
    }

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        doPost(request, response);
    }

    private String runPythonPrediction(JSONObject inputJson) {
        try {
            String pythonExecutable = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3";
            String scriptPath = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/predict.py";

            File scriptFile = new File(scriptPath);
            if (!scriptFile.exists()) {
                JSONObject error = new JSONObject();
                error.put("error", "Python 스크립트를 찾을 수 없습니다.");
                return error.toString();
            }

            ProcessBuilder pb = new ProcessBuilder(pythonExecutable, scriptPath);
            pb.redirectErrorStream(true);  // stderr도 stdout으로 함께 읽음
            Process process = pb.start();

            // ✅ JSON 입력값을 stdin으로 전달
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            writer.write(inputJson.toString());
            writer.flush();
            writer.close();

            // ✅ stdout 결과 읽기
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            String fullOutput = output.toString().trim();

            // ✅ Python stdout 디버깅 출력
            System.err.println("[Python output start]");
            System.err.println(fullOutput);
            System.err.println("[Python output end]");

            // ✅ JSON 블록만 추출
            int start = fullOutput.indexOf("{");
            int end = fullOutput.lastIndexOf("}") + 1;

            if (start == -1 || end == -1 || start >= end) {
                JSONObject error = new JSONObject();
                error.put("error", "Python 결과에서 유효한 JSON을 추출하지 못했습니다.");
                error.put("raw_output", fullOutput);
                return error.toString();
            }

            String jsonOnly = fullOutput.substring(start, end);

            // ✅ JSON 유효성 검증
            new JSONObject(jsonOnly);
            return jsonOnly;

        } catch (Exception e) {
            e.printStackTrace();
            JSONObject error = new JSONObject();
            error.put("error", "서블릿 예외 발생: " + e.toString());
            return error.toString();
        }
    }
}