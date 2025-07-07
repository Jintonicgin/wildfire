package com.seed.util;

import java.security.MessageDigest;

public class SHA256 {
  public static String encrypt(String input) {
    try {
      MessageDigest md = MessageDigest.getInstance("SHA-256");
      byte[] bytes = md.digest(input.getBytes("UTF-8"));
      StringBuilder sb = new StringBuilder();
      byte b;
      int i;
      byte[] arrayOfByte1;
      for (i = (arrayOfByte1 = bytes).length, b = 0; b < i; ) {
        byte b1 = arrayOfByte1[b];
        sb.append(String.format("%02x", new Object[] { Byte.valueOf(b1) }));
        b++;
      } 
      return sb.toString();
    } catch (Exception e) {
      throw new RuntimeException("SHA-256 암호화 오류", e);
    } 
  }
}
