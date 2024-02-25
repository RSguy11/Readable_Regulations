document.getElementById("submit-button").addEventListener("click",submitForm);



document.getElementById("input-results-box").addEventListener("click", function(event) {
    if (event.target.id === "label") {
        switchPage();
    }
    if (event.target.id === "box-click"){
        expantion();
    }
});

// funny music :)
function playMusic(){
    var music = new Audio('kachow.mp3');
    music.play();
    }

document.getElementById("nav-bar").addEventListener("click", function(event) {
    if (event.target.id === "economic") {
        ClickedHighlight("economic");
    }
    if (event.target.id === "defence"){
        ClickedHighlight("defence");
    }
    if (event.target.id === "environmental"){
        ClickedHighlight("environmental");
    }
    if (event.target.id === "transportation"){
        ClickedHighlight("transportation");
    }
    if (event.target.id === "other"){
        ClickedHighlight("other");
    }
});



function ClickedHighlight(Eid){
    var ele1 = document.getElementById(Eid);
    
    if (ele1.classList.contains("clickedHighlight")){
        ele1.classList.remove("clickedHighlight");
    }else{
        ele1.classList.add("clickedHighlight");

    }
}


// function test(){
//     document.getElementById("three").innerText = "hello";
// }

function switchPage(){
    window.location.href ='AdvancedPage.html';
}

function expantion(){
    var inputbox = document.getElementById("label-");

}


function submitForm() {
    var inputbox = document.getElementById("input-results-box");


    // Get the value from the input field
    var linkValue = document.getElementById("link-ins").value;
    var box = document.createElement("div");
    box.classList.add("box")
    box.id = "box-click";

    var label = document.createElement("label");
    label.innerText = "Submitted Link: " + linkValue;
    label.classList.add("box-inner-text");
    label.id = "label"

    box.appendChild(label);

    inputbox.appendChild(box);

    // document.getElementById("three").innerText = box.id;

    

}


