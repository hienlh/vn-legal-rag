const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const diagramsDir = path.join(__dirname, '..', 'diagrams');
const outputDir = path.join(diagramsDir, 'output');

const figures = [
  'figure2-1-document-tree',
  'figure2-2-kg-subgraph',
  'figure3-1-system-architecture',
  'figure3-2-tree-traversal',
  'figure3-3-end-to-end-flow',
  'figure3-4-query-analysis',
  'figure3-5-tier2-dual-level',
  'figure3-6-offline-pipeline',
];

function buildViewerHtml(drawioXml) {
  const b64 = Buffer.from(drawioXml, 'utf-8').toString('base64');
  return `<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  body { margin: 0; padding: 20px; background: white; }
  .mxgraph { max-width: 100%; }
</style>
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
</script>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body></html>`;
}

(async () => {
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  const browser = await puppeteer.launch({ headless: 'new' });

  for (const name of figures) {
    const drawioPath = path.join(diagramsDir, `${name}.drawio`);
    const pngPath = path.join(outputDir, `${name}-drawio.png`);

    if (!fs.existsSync(drawioPath)) {
      console.log(`SKIP: ${name}.drawio not found`);
      continue;
    }

    const xml = fs.readFileSync(drawioPath, 'utf-8');
    const html = buildViewerHtml(xml);

    const tmpHtml = path.join(outputDir, `_tmp_${name}.html`);
    fs.writeFileSync(tmpHtml, html, 'utf-8');

    const page = await browser.newPage();
    await page.setViewport({ width: 1800, height: 1200, deviceScaleFactor: 2 });

    try {
      await page.goto(`file:///${tmpHtml.replace(/\\/g, '/')}`, {
        waitUntil: 'networkidle0',
        timeout: 30000,
      });

      await page.waitForSelector('.mxgraph svg', { timeout: 15000 });
      await new Promise(r => setTimeout(r, 1000));

      const svg = await page.$('.mxgraph svg');
      const box = await svg.boundingBox();

      await page.setViewport({
        width: Math.ceil(box.x + box.width + 40),
        height: Math.ceil(box.y + box.height + 40),
        deviceScaleFactor: 2,
      });

      await page.goto(`file:///${tmpHtml.replace(/\\/g, '/')}`, {
        waitUntil: 'networkidle0',
        timeout: 30000,
      });
      await page.waitForSelector('.mxgraph svg', { timeout: 15000 });
      await new Promise(r => setTimeout(r, 1000));

      await page.screenshot({ path: pngPath, fullPage: true, omitBackground: false });
      const newBox = await (await page.$('.mxgraph svg')).boundingBox();
      console.log(`OK: ${name}.drawio -> PNG (${Math.ceil(newBox.width)}x${Math.ceil(newBox.height)})`);
    } catch (err) {
      console.log(`FAIL: ${name}.drawio -> ${err.message}`);
    }

    await page.close();
    try { fs.unlinkSync(tmpHtml); } catch {}
  }

  await browser.close();
  console.log('All done.');
})();
