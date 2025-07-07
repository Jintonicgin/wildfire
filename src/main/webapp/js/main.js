document.addEventListener("DOMContentLoaded", () => {
  const textElement = document.getElementById("type-text");
  const subElement = document.getElementById("type-sub");
  const enterBtn = document.getElementById("enterBtn");

  const mainText = "SEED";
  const subText = "Smart Ecological Evaluation & Diagnostics";

  let mainIndex = 0;
  let subIndex = 0;

  textElement.classList.add("typing-cursor");

  function typeMain() {
    if (mainIndex < mainText.length) {
      textElement.textContent += mainText.charAt(mainIndex);
      mainIndex++;
      setTimeout(typeMain, 160);
    } else {
 
      textElement.classList.remove("typing-cursor");
      textElement.style.borderRight = "none"; 
      void textElement.offsetWidth;

      subElement.style.visibility = "visible";
      subElement.classList.add("typing-cursor");
      setTimeout(typeSub, 400);
    }
  }

  function typeSub() {
    if (subIndex < subText.length) {
      subElement.textContent += subText.charAt(subIndex);
      subIndex++;
      setTimeout(typeSub, 35);
    } else {
      subElement.classList.remove("typing-cursor");
      subElement.style.borderRight = "none";
      void subElement.offsetWidth;

      setTimeout(() => {
        enterBtn.style.display = "inline-block";
      }, 400);
    }
  }

  typeMain();

  enterBtn.addEventListener("click", () => {
    document.querySelector(".intro-screen").style.display = "none";
    document.querySelector(".main-content").style.display = "block";
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const scrollElements = document.querySelectorAll(".scroll-fade");

  const scrollInView = (el) => {
    const rect = el.getBoundingClientRect();
    return rect.top <= window.innerHeight * 0.8;
  };

  const revealOnScroll = () => {
    scrollElements.forEach((el) => {
      if (scrollInView(el)) {
        el.classList.add("reveal");
      }
    });
  };

  window.addEventListener("scroll", revealOnScroll);
  revealOnScroll();
});