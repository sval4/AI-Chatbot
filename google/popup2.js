// popup.js

const extensionURL2 = chrome.runtime.getURL('all.js');
const script2 = document.createElement('script');
script2.src = extensionURL2;
document.head.appendChild(script2);

// Rest of your JavaScript code can go here
