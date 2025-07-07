document.addEventListener("DOMContentLoaded", function () {
  const citySelect = document.getElementById("city");
  const dongSelect = document.getElementById("dong");

  const cityTS = new TomSelect(citySelect, {
    placeholder: "시/군 선택",
    copyClassesToDropdown: true,
    create: false,
    allowEmptyOption: true,
    controlInput: false,
    dropdownParent: "body",
    dropdownClass: 'ts-dropdown',
    controlClass: 'ts-control',
    render: {
      option: function (data, escape) {
        return `<div class="option">${escape(data.text)}</div>`;
      }
    }
  });

  const confirmBtn = document.getElementById("confirmBtn");
  const resultText = document.getElementById("selectedRegion");

  confirmBtn.addEventListener("click", function () {
    const city = citySelect.value;

    if (!city) {
      resultText.textContent = "⚠️ 시/군을 선택해주세요.";
      resultText.style.color = "red";
      return;
    }
  });
});