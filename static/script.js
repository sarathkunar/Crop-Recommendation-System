console.log("Smart Crop Recommendation System Loaded");

document.addEventListener("DOMContentLoaded", function(){

    const form = document.querySelector("form");

    form.addEventListener("submit", function(){

        const button = document.querySelector("button");

        button.innerHTML = "Predicting...";

        button.style.opacity = "0.8";

    });

});