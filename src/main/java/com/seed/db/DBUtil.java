package com.seed.db;

import java.sql.Connection;
import java.sql.DriverManager;

public class DBUtil {
    public static Connection getConnection() throws Exception {
        String url = "jdbc:oracle:thin:@localhost:1521:xe";
        String user = "seed";
        String password = "seed1234";
        Class.forName("oracle.jdbc.OracleDriver");
        return DriverManager.getConnection(url, user, password);
    }
}