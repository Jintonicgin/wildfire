package com.seed.model;

public class Member {
  private String username;
  
  private String password;
  
  private String email;
  
  private String name;
  
  public Member(String username, String password, String email, String name) {
    this.username = username;
    this.password = password;
    this.email = email;
    this.name = name;
  }
  
  public String getUsername() {
    return this.username;
  }
  
  public String getPassword() {
    return this.password;
  }
  
  public String getEmail() {
    return this.email;
  }
  
  public String getName() {
    return this.name;
  }
}
