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