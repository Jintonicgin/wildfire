package com.seed.dao;

import com.seed.model.Member;
import com.seed.db.DBUtil;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

public class MemberDAO {

    // 회원 등록
    public boolean insertMember(Member member) {
        String sql = "INSERT INTO MEMBER (USERNAME, PASSWORD, EMAIL, NAME) VALUES (?, ?, ?, ?)";
        try (Connection conn = DBUtil.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, member.getUsername());
            pstmt.setString(2, member.getPassword());
            pstmt.setString(3, member.getEmail());
            pstmt.setString(4, member.getName());

            int rows = pstmt.executeUpdate();
            return rows > 0;

        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    // 로그인 유효성 검사
    public boolean validateLogin(String username, String password) {
        String sql = "SELECT * FROM MEMBER WHERE USERNAME = ? AND PASSWORD = ?";
        try (Connection conn = DBUtil.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, username);
            pstmt.setString(2, password);

            ResultSet rs = pstmt.executeQuery();
            return rs.next(); // 값이 존재하면 로그인 성공

        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    // 아이디 중복 확인
    public boolean isUsernameTaken(String username) {
        String sql = "SELECT * FROM MEMBER WHERE USERNAME = ?";
        try (Connection conn = DBUtil.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, username);
            ResultSet rs = pstmt.executeQuery();
            return rs.next();

        } catch (Exception e) {
            e.printStackTrace();
            return true; // 오류 발생 시 중복된 것으로 간주
        }
    }

    // 이메일 중복 확인
    public boolean isEmailTaken(String email) {
        String sql = "SELECT * FROM MEMBER WHERE EMAIL = ?";
        try (Connection conn = DBUtil.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, email);
            ResultSet rs = pstmt.executeQuery();
            return rs.next();

        } catch (Exception e) {
            e.printStackTrace();
            return true;
        }
    }

    // 이름 + 이메일로 아이디 찾기
    public String findUsernameByNameAndEmail(String name, String email) {
        String sql = "SELECT USERNAME FROM MEMBER WHERE NAME = ? AND EMAIL = ?";
        try (Connection conn = DBUtil.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, name);
            pstmt.setString(2, email);
            ResultSet rs = pstmt.executeQuery();
            if (rs.next()) {
                return rs.getString("USERNAME");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    // 아이디 + 이메일 일치 확인 (비번 찾기용)
    public boolean checkUserByIdAndEmail(String username, String email) {
        String sql = "SELECT * FROM MEMBER WHERE USERNAME = ? AND EMAIL = ?";
        try (Connection conn = DBUtil.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, username);
            pstmt.setString(2, email);
            ResultSet rs = pstmt.executeQuery();
            return rs.next();

        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    // 비밀번호 재설정
    public boolean updatePassword(String username, String newPw) {
        String sql = "UPDATE MEMBER SET PASSWORD = ? WHERE USERNAME = ?";
        try (Connection conn = DBUtil.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, newPw);
            pstmt.setString(2, username);
            int rows = pstmt.executeUpdate();
            return rows > 0;

        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}