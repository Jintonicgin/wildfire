package com.seed.servlet;

import org.json.JSONObject;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.*;
import java.sql.*;

@WebServlet("/PredictionServlet")
public class PredictionServlet extends HttpServlet {

    private static final long serialVersionUID = 1L;

    private static final String ORACLE_URL = "jdbc:oracle:thin:@localhost:1521:xe";
    private static final String ORACLE_USER = "wildfire";
    private static final String ORACLE_PASS = "1234";
    private static final String PYTHON_EXE = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3";
    private static final String SCRIPT_PATH = "/Users/mmymacymac/Developer/Projects/eclipse/WildFire/dataset/";
    private static final String SCRIPT_REALTIME = SCRIPT_PATH + "predict.py";
    private static final String SCRIPT_FROM_DB = SCRIPT_PATH + "predict_from_feature.py";

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        request.setCharacterEncoding("UTF-8");
        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");

        PrintWriter out = response.getWriter();

        try {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = request.getReader().readLine()) != null) {
                sb.append(line);
            }
            JSONObject jsonRequest = new JSONObject(sb.toString());

            String scriptToRun;
            JSONObject scriptInput;

            if (jsonRequest.has("city_name")) {
                String city = jsonRequest.getString("city_name");
                JSONObject featuresFromDB = loadFeaturesFromDB(city);

                if (featuresFromDB == null) {
                    response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                    out.print(new JSONObject().put("error", "DB에서 해당 도시 데이터를 찾을 수 없습니다. (Table: REGION_PREDICTION_FEATURES)"));
                    return;
                }
                
                featuresFromDB.put("durationHours", jsonRequest.getInt("durationHours"));
                featuresFromDB.put("timestamp", jsonRequest.getString("timestamp"));
                
                scriptToRun = SCRIPT_FROM_DB;
                scriptInput = featuresFromDB;

            } else if (jsonRequest.has("latitude") && jsonRequest.has("longitude")) {
                scriptToRun = SCRIPT_REALTIME;
                scriptInput = jsonRequest;

            } else {
                response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                out.print(new JSONObject().put("error", "요청에 필요한 정보(city_name 또는 lat/lng)가 없습니다."));
                return;
            }

            String result = runPythonScript(scriptToRun, scriptInput);
            out.print(result);

        } catch (Exception e) {
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            out.print(new JSONObject().put("error", "서버 처리 중 예외 발생: " + e.toString()));
            e.printStackTrace();
        } finally {
            out.flush();
            out.close();
        }
    }

    private JSONObject loadFeaturesFromDB(String city) {
        try {
            Class.forName("oracle.jdbc.driver.OracleDriver");
            try (Connection conn = DriverManager.getConnection(ORACLE_URL, ORACLE_USER, ORACLE_PASS)) {
                String sql = "SELECT * FROM REGION_PREDICTION_FEATURES WHERE REGION_NAME = ?";
                try (PreparedStatement stmt = conn.prepareStatement(sql)) {
                    stmt.setString(1, city);
                    try (ResultSet rs = stmt.executeQuery()) {
                        if (rs.next()) {
                            ResultSetMetaData meta = rs.getMetaData();
                            JSONObject obj = new JSONObject();
                            for (int i = 1; i <= meta.getColumnCount(); i++) {
                                String key = meta.getColumnName(i).toLowerCase();
                                Object val = rs.getObject(i);
                                if (val == null) {
                                    obj.put(key, JSONObject.NULL);
                                } else if (val instanceof Number || val instanceof Boolean) {
                                    obj.put(key, val);
                                } else {
                                    obj.put(key, val.toString());
                                }
                            }
                            return obj;
                        }
                    }
                }
            }
        } catch (SQLException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    private String runPythonScript(String scriptPath, JSONObject inputJson) throws Exception {
        ProcessBuilder pb = new ProcessBuilder(PYTHON_EXE, scriptPath);
        pb.redirectErrorStream(false);
        Process process = pb.start();

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
            writer.write(inputJson.toString());
            writer.flush(); // ✅ 꼭 flush 해야 Python이 입력을 읽음
        }

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line);
            }
        }

        StringBuilder errorOutput = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                errorOutput.append(line);
            }
        }

        int exitCode = process.waitFor();

        // ✅ 디버깅 출력 추가
        System.out.println("Python exit code: " + exitCode);
        System.out.println("Python stdout: " + output.toString());
        System.out.println("Python stderr: " + errorOutput.toString());

        if (output.length() == 0) {
            // 빈 출력일 경우 클라이언트에 명시적으로 전달
            return new JSONObject()
                .put("error", "Python 실행 결과가 비어 있습니다.")
                .put("stderr", errorOutput.toString())
                .toString();
        }

        return output.toString();
    }
    
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        doPost(request, response);
    }
}