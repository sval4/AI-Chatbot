
// Utils
function get(selector, root = document) {
    return root.querySelector(selector);
}

const msgerForm = get(".msger-inputarea");
const msgerInput = get(".msger-input");
const msgerChat = get(".msger-chat");
const addButton = get("#add-button");


// Icons made by Freepik from www.flaticon.com
const BOT_IMG = "static/styles/chatbot-pfp.png";
const PERSON_IMG = "static/styles/human-pfp.jpeg";
const BOT_NAME = "ChatBot";
const PERSON_NAME = "You";


addButton.addEventListener("click", event =>{
  event.preventDefault();

  const msgText = msgerInput.value;
  if (!msgText) return;

  msgerInput.disabled = true;
  msgerInput.value = "";
  botResponse2(msgText);
});

msgerForm.addEventListener("submit", event => {
  event.preventDefault();
  
  const msgText = msgerInput.value;
  if (!msgText) return;

  msgerInput.disabled = true;

  appendMessageHuman(PERSON_NAME, PERSON_IMG, "right", msgText);
  msgerInput.value = "";
  botResponse(msgText);
});

function appendMessage(name, img, side, answer, source) {
  //   Simple solution for small apps
  const sourceSection = source ? `<br><div class="msg-text">Source: <a href=${source} target="_blank">${source}</a></div>` : '';
  const msgHTML = `
<div class="msg ${side}-msg">
<div class="msg-img" style="background-image: url(${img});"></div>

<div class="msg-bubble">
<div class="msg-info">
  <div class="msg-info-name">${name}</div>
  <div class="msg-info-time">${formatDate(new Date())}</div>
</div>

<div class="msg-text">${answer}</div>
${sourceSection}
</div>
</div>
`;


  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop += 500;
}

function appendMessageHuman(name, img, side, text) {
  //   Simple solution for small apps
  const msgHTML = `
<div class="msg ${side}-msg">
<div class="msg-img" style="background-image: url(${img});"></div>

<div class="msg-bubble">
<div class="msg-info">
  <div class="msg-info-name">${name}</div>
  <div class="msg-info-time">${formatDate(new Date())}</div>
</div>

<div class="msg-text">${text}</div>
</div>
</div>
`;


  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop += 500;
}

function botResponse(rawText) {
    // Bot Response
    fetch(`http://127.0.0.1:5000/get?msg=${encodeURIComponent(rawText)}`, {
      method: "GET",
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Network response was not ok (Status: ${response.status})`);
        }
        return response.json();
      })
      .then((data) => {
        console.log(rawText);
        console.log(data);
        const answer = data["answer"];
        const source = data["source"];
        appendMessage(BOT_NAME, BOT_IMG, "left", answer, source);
        msgerInput.disabled = false;
      })
      .catch((error) => {
        console.error("Error:", error);
        // You can handle specific errors or log more details here
      });
  }
  
  

function botResponse2(rawText) {
    // Bot Response
    fetch(`http://127.0.0.1:5000/get2?msg=${encodeURIComponent(rawText)}`, {
      method: "GET",
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        console.log(rawText);
        console.log(data);
        appendMessage(BOT_NAME, BOT_IMG, "left", data["answer"], "");
        msgerInput.disabled = false;
      })
      .catch((error) => {
        console.error("Error:", error);
        // Handle errors here if needed
      });
  }
  




function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();

  return `${h.slice(-2)}:${m.slice(-2)}`;
}

const time = get("time-now");

document.querySelector('.msg-info-time').innerHTML = formatDate(new Date());