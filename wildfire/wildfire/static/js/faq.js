document.addEventListener('DOMContentLoaded', function() {
    const faqQuestions = document.querySelectorAll('.faq-question');

    faqQuestions.forEach(question => {
        question.addEventListener('click', function() {
            const answer = this.nextElementSibling; // Get the next sibling, which should be the answer
            if (answer) {
                answer.classList.toggle('active');
            }
        });
    });
});