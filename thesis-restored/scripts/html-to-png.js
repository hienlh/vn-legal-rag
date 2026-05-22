const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const graphicsDir = path.join(__dirname, '..', 'graphics');

const figures = [
  { html: 'figure2-1-document-tree.html', png: 'figure2-1-document-tree.png' },
  { html: 'figure2-2-kg-subgraph.html', png: 'figure2-2-kg-subgraph.png' },
  { html: 'figure3-1-system-architecture.html', png: 'figure3-1-system-architecture.png' },
  { html: 'figure3-2-tree-traversal.html', png: 'figure3-2-tree-traversal.png' },
  { html: 'figure3-3-end-to-end-flow.html', png: 'figure3-3-end-to-end-flow.png' },
];

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });

  for (const fig of figures) {
    const htmlPath = path.join(graphicsDir, fig.html);
    const pngPath = path.join(graphicsDir, fig.png);

    if (!fs.existsSync(htmlPath)) {
      console.log(`SKIP: ${fig.html} not found`);
      continue;
    }

    const page = await browser.newPage();
    await page.goto(`file:///${htmlPath.replace(/\\/g, '/')}`, { waitUntil: 'networkidle0' });

    const body = await page.$('body');
    const box = await body.boundingBox();

    await page.setViewport({
      width: Math.ceil(box.width),
      height: Math.ceil(box.height),
      deviceScaleFactor: 2,
    });

    await page.goto(`file:///${htmlPath.replace(/\\/g, '/')}`, { waitUntil: 'networkidle0' });

    await page.screenshot({ path: pngPath, fullPage: true, omitBackground: false });
    console.log(`OK: ${fig.html} -> ${fig.png}`);
    await page.close();
  }

  await browser.close();
  console.log('All done.');
})();
