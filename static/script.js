document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("cropForm");
    const button = document.getElementById("predictBtn");

    if (form && button) {
        form.addEventListener("submit", function (e) {
            if (!form.checkValidity()) {
                return;
            }
            button.disabled = true;
            button.innerText = "⏳ Analyzing Soil & Weather...";
            form.submit();
        });
    }
});