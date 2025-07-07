document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("#signupForm");

  form.addEventListener("submit", async function (e) {
    const username = form.querySelector("input[name='username']").value.trim();
    const password = form.querySelector("input[name='password']").value;
    const confirmPassword = form.querySelector("input[name='confirmPassword']").value;
    const email = form.querySelector("input[name='email']").value.trim();
    const agree = form.querySelector("input[name='agree']");

   
    const idPattern = /^[A-Za-z0-9]{4,20}$/;
    if (!idPattern.test(username)) {
      alert("아이디는 영문자와 숫자만 사용하며 4~20자여야 합니다.");
      e.preventDefault();
      return;
    }

    
    const idCheck = await fetch(`/WildFire/api/check-username?username=${encodeURIComponent(username)}`);
    const idResult = await idCheck.text();
    if (idResult === "duplicate") {
      alert("중복된 아이디입니다. 다른 아이디를 사용해주세요.");
      e.preventDefault();
      return;
    }

   
    const pwPattern = /^(?=.*[!@#$%^&*(),.?":{}|<>])[A-Za-z\d!@#$%^&*(),.?":{}|<>]{8,}$/;
    if (!pwPattern.test(password)) {
      alert("비밀번호는 8자 이상이며 특수문자를 최소 1개 포함해야 합니다.");
      e.preventDefault();
      return;
    }

    if (password !== confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      e.preventDefault();
      return;
    }

  
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(email)) {
      alert("올바른 이메일 형식을 입력해주세요.");
      e.preventDefault();
      return;
    }

 
    const emailCheck = await fetch(`/WildFire/api/check-email?email=${encodeURIComponent(email)}`);
    const emailResult = await emailCheck.text();
    if (emailResult === "duplicate") {
      alert("이미 가입된 이메일입니다.");
      e.preventDefault();
      return;
    }

    // ✅ 유효성 통과 → submit 진행
  });
});