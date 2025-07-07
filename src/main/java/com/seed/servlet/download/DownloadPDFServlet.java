package com.seed.servlet.download;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;
import java.io.*;
import java.util.Base64;
import org.apache.pdfbox.pdmodel.*;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDType0Font;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;

@WebServlet("/DownloadPDFServlet")
public class DownloadPDFServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        request.setCharacterEncoding("UTF-8");
        response.setContentType("application/pdf");
        response.setHeader("Content-Disposition", "attachment; filename=\"ndvi_report.pdf\"");

        HttpSession session = request.getSession(false);
        String user = (session != null) ? (String) session.getAttribute("user") : null;
        if (user == null) {
            response.sendRedirect("/FireWild/jsp/login.jsp");
            return;
        }

        // JSON 파싱
        StringBuilder jsonBuilder = new StringBuilder();
        try (BufferedReader reader = request.getReader()) {
            String line;
            while ((line = reader.readLine()) != null) {
                jsonBuilder.append(line);
            }
        }
        String json = jsonBuilder.toString();
        String region = extractValue(json, "region");
        String ndviBase64 = extractBase64(json, "ndvi");
        String barBase64 = extractBase64(json, "bar");
        String ndviStartStr = extractValue(json, "ndviStart");
        String ndviEndStr = extractValue(json, "ndviEnd");
        System.out.println("region = " + region);
        System.out.println("ndvi.length = " + (ndviBase64 != null ? ndviBase64.length() : "null"));
        System.out.println("bar.length = " + (barBase64 != null ? barBase64.length() : "null"));
        String[] varLabels = extractArray(json, "varLabels");
        String[] varValuesStr = extractArray(json, "varValues");

        float ndviStart = 0.0f;
        float ndviEnd = 0.0f;

        try {
            ndviStart = Float.parseFloat(ndviStartStr);
            ndviEnd = Float.parseFloat(ndviEndStr);
        } catch (NumberFormatException e) {
            System.err.println("NDVI 수치 파싱 실패: " + e.getMessage());
        }
        
        int[] varValues = new int[varValuesStr.length];
        for (int i = 0; i < varValuesStr.length; i++) {
            try {
                varValues[i] = (int) Float.parseFloat(varValuesStr[i].trim());
            } catch (NumberFormatException e) {
                varValues[i] = 0; // 비어있거나 잘못된 값일 경우 기본값 0
            }
        }

        String ndviDesc = String.format("→ 산불 직후 NDVI 값은 %.2f이며, 24개월 후 %.2f까지 회복되었습니다.", ndviStart, ndviEnd);
        String barDesc = String.format("→ %s(%d%%), %s(%d%%), %s(%d%%) 순으로 회복에 영향을 주었습니다.",
                varLabels[0], varValues[0], varLabels[1], varValues[1], varLabels[2], varValues[2]);

        try (PDDocument document = new PDDocument()) {
            InputStream fontStream = getServletContext().getResourceAsStream("/fonts/NanumGothic.ttf");
            if (fontStream == null) throw new FileNotFoundException("폰트 파일을 찾을 수 없습니다.");
            PDType0Font font = PDType0Font.load(document, fontStream);

            PDPage page = new PDPage(PDRectangle.A4);
            document.addPage(page);

            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                float margin = 50;
                float width = page.getMediaBox().getWidth() - 2 * margin;

                content.beginText();
                content.setFont(font, 14);
                content.newLineAtOffset(margin, 770); // ✅ 제목 위치
                content.showText(region + " 산불 이후 NDVI 변화율");
                content.newLineAtOffset(0, -20);       // ↓ 이 위치 다음 텍스트가 내려감
                content.setFont(font, 12);
                content.showText("아래 그래프는 " + region + "의 NDVI 변화를 나타냅니다.");
                content.endText();

                content.beginText();
                content.setFont(font, 11);
                // ❌ 여기가 745로 너무 겹쳐 있음 → 수정 필요
                content.newLineAtOffset(margin, 715); // ✅ 745 → 715 정도로 낮춤
                content.showText(ndviDesc);
                content.endText();

                if (ndviBase64 != null && !ndviBase64.isEmpty()) {
                    PDImageXObject ndviImage = PDImageXObject.createFromByteArray(document, decodeBase64Image(ndviBase64), "ndvi");
                    content.drawImage(ndviImage, margin, 580, width, 110);
                }

                content.beginText();
                content.setFont(font, 14);
                content.newLineAtOffset(margin, 490);
                content.showText(region + " NDVI 회복 주요 변수 영향력 (%)");
                content.endText();

                content.beginText();
                content.setFont(font, 11);
                content.newLineAtOffset(margin, 470);
                content.showText(barDesc);
                content.endText();

                if (barBase64 != null && !barBase64.isEmpty()) {
                    PDImageXObject barImage = PDImageXObject.createFromByteArray(document, decodeBase64Image(barBase64), "bar");
                    float barWidth = width * 0.45f;
                    float barHeight = 420;
                    float x = margin + (width - barWidth) / 2;
                    float y = 40;
                    content.drawImage(barImage, x, y, barWidth, barHeight);
                }
            }

            document.save(response.getOutputStream());
            response.flushBuffer();

        } catch (Exception e) {
            e.printStackTrace();
            response.setContentType("text/plain;charset=UTF-8");
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            response.getWriter().write("PDF 생성 중 오류 발생: " + e.getMessage());
        }
    }

    private String extractBase64(String json, String key) {
        String marker = "\"" + key + "\":\"data:image/png;base64,";
        int start = json.indexOf(marker);
        if (start == -1) return null;
        start += marker.length();
        int end = json.indexOf("\"", start);
        if (end == -1) return null;
        return json.substring(start, end).replaceAll("\\\\n", "").replaceAll("\\\\", "");
    }

    private String extractValue(String json, String key) {
        String marker = "\"" + key + "\":\"";
        int start = json.indexOf(marker);
        if (start == -1) return "";
        start += marker.length();
        int end = json.indexOf("\"", start);
        if (end == -1) return "";
        return json.substring(start, end);
    }

    private String[] extractArray(String json, String key) {
        String marker = "\"" + key + "\":[";
        int start = json.indexOf(marker);
        if (start == -1) return new String[0];
        start += marker.length();
        int end = json.indexOf("]", start);
        if (end == -1) return new String[0];
        String raw = json.substring(start, end).replaceAll("\"", "").trim();
        return raw.split(",");
    }

    private byte[] decodeBase64Image(String base64Data) throws IllegalArgumentException {
        if (base64Data == null || base64Data.isEmpty()) {
            throw new IllegalArgumentException("이미지 데이터가 비어 있습니다.");
        }
        return Base64.getDecoder().decode(base64Data);
    }
}