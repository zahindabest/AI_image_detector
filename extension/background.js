// 1. Create the Context Menu on Install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkDeepFake",
    title: "🔍 Scan for DeepFake",
    contexts: ["image"]
  });
});

// 2. Handle the Click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "checkDeepFake") {
    
    // Notify user that scanning started
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => alert("Scanning image... Please wait.")
    });

    try {
      const imageUrl = info.srcUrl;
      
      // A. Fetch the image data from the URL
      const response = await fetch(imageUrl);
      const blob = await response.blob();

      // B. Prepare the form data for the API
      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      // C. Send to Local API
      // Make sure Uvicorn server is running!
      const apiResponse = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
      });

      const data = await apiResponse.json();

      // D. Format the Hit or Miss Report
      if (data.error) {
        showResult(tab.id, `Error: ${data.error}`);
      } else {
        // Build a detailed info
        const message = `
ANALYSIS RESULT: ${data.label}
--------------------------------
Confidence: ${data.confidence}%
Model Used: ${data.router_decision.model_used}
Sharpness: ${data.router_decision.sharpness_score} (Threshold: ${data.router_decision.threshold})

${data.label === "FAKE" ? "This image shows signs of AI generation." : "This image appears natural."}
        `;
        
        showResult(tab.id, message);
      }

    } catch (error) {
      console.error("Error:", error);
      showResult(tab.id, "Connection Failed. Is the python server running?");
    }
  }
});

// Helper function to show alerts in the active tab
function showResult(tabId, message) {
  chrome.scripting.executeScript({
    target: { tabId: tabId },
    func: (msg) => alert(msg),
    args: [message]
  });
}