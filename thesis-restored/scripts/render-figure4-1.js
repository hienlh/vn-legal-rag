const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const diagramsDir = path.join(__dirname, '..', 'diagrams');
const outputDir = path.join(diagramsDir, 'output');

function buildViewerHtml(drawioXml) {
  const b64 = Buffer.from(drawioXml, 'utf-8').toString('base64');
  return `<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>body { margin: 0; padding: 20px; background: white; } .mxgraph { max-width: 100%; }</style>
</head><body>
<div id="target"></div>
<script>
  var raw = atob("${b64}");
  var bytes = new Uint8Array(raw.length);
  for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  var xml = new TextDecoder("utf-8").decode(bytes);
  var div = document.getElementById("target");
  div.className = "mxgraph";
  div.setAttribute("data-mxgraph", JSON.stringify({highlight:"none",nav:false,resize:false,toolbar:"",edit:"",xml:xml}));
<\/script>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"><\/script>
</body></html>`;
}

(async () => {
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  const name = 'figure4-1-deployment-architecture';
  const drawioPath = path.join(diagramsDir, name + '.drawio');
  const pngPath = path.join(outputDir, name + '-drawio.png');

  if (!fs.existsSync(drawioPath)) {
    console.error('ERROR: drawio file not found:', drawioPath);
    process.exit(1);
  }

  const xml = fs.readFileSync(drawioPath, 'utf-8');
  const html = buildViewerHtml(xml);

  const tmpHtml = path.join(outputDir, '_tmp_' + name + '.html');
  fs.writeFileSync(tmpHtml, html, 'utf-8');

  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1800, height: 1200, deviceScaleFactor: 2 });

  const fileUrl = 'file:///' + tmpHtml.replace(/\\/g, '/');

  await page.goto(fileUrl, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.waitForSelector('.mxgraph svg', { timeout: 15000 });
  await new Promise(r => setTimeout(r, 1500));

  const svg = await page.$('.mxgraph svg');
  const box = await svg.boundingBox();

  await page.setViewport({
    width: Math.ceil(box.x + box.width + 40),
    height: Math.ceil(box.y + box.height + 40),
    deviceScaleFactor: 2,
  });

  await page.goto(fileUrl, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.waitForSelector('.mxgraph svg', { timeout: 15000 });
  await new Promise(r => setTimeout(r, 1500));

  await page.screenshot({ path: pngPath, fullPage: true, omitBackground: false });
  const newSvg = await page.$('.mxgraph svg');
  const newBox = await newSvg.boundingBox();
  console.log('OK: ' + name + ' -> ' + pngPath + ' (' + Math.ceil(newBox.width) + 'x' + Math.ceil(newBox.height) + ')');

  await page.close();
  await browser.close();
  try { fs.unlinkSync(tmpHtml); } catch (_) {}
})().catch(e => {
  console.error('FAIL:', e.message);
  process.exit(1);
});
