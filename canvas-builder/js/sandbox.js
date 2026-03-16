const PLACEHOLDER_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Canvas</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #0f0f0f;
    color: #888;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    gap: 24px;
  }
  .icon { font-size: 48px; opacity: 0.4; }
  h2 { font-size: 18px; font-weight: 400; color: #555; }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    max-width: 480px;
  }
  .chip {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    color: #666;
    cursor: pointer;
    transition: all 0.15s;
  }
  .chip:hover {
    border-color: #444;
    color: #aaa;
    background: #222;
  }
</style>
</head>
<body>
  <div class="icon">⬜</div>
  <h2>Type a prompt to start building</h2>
  <div class="chips" id="chips">
    <div class="chip" onclick="parent.postMessage({type:'chip',text:'make a to-do app'},'*')">to-do app</div>
    <div class="chip" onclick="parent.postMessage({type:'chip',text:'build a landing page for a startup'},'*')">landing page</div>
    <div class="chip" onclick="parent.postMessage({type:'chip',text:'create an analytics dashboard'},'*')">analytics dashboard</div>
    <div class="chip" onclick="parent.postMessage({type:'chip',text:'make a calculator'},'*')">calculator</div>
    <div class="chip" onclick="parent.postMessage({type:'chip',text:'build a markdown editor with live preview'},'*')">markdown editor</div>
    <div class="chip" onclick="parent.postMessage({type:'chip',text:'create a pomodoro timer'},'*')">pomodoro timer</div>
  </div>
</body>
</html>`;

function renderHtml(iframeEl, htmlString) {
  return new Promise((resolve) => {
    iframeEl.addEventListener('load', resolve, { once: true });
    iframeEl.srcdoc = htmlString;
  });
}

function showPlaceholder(iframeEl) {
  return renderHtml(iframeEl, PLACEHOLDER_HTML);
}

function getHtml(iframeEl) {
  return iframeEl.srcdoc || '';
}
