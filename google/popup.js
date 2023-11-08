// popup.js

const extensionURL = chrome.runtime.getURL('jquery.min.js');
const script = document.createElement('script');
script.src = extensionURL;
document.head.appendChild(script);

// Rest of your JavaScript code can go here
