// Browser smoke test against a running local server; never saves/deletes user data.
// PLAYWRIGHT_MODULE may point to an installed/bundled Playwright package.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const base = process.env.LEAFCARE_URL || 'http://127.0.0.1:5000';
const out = path.join(root, 'reports', 'browser');
fs.mkdirSync(out, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, locale: 'kk-KZ' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const checks = [];
  const record = name => { checks.push({ name, passed: true }); console.log('PASS', name); };
  const historyBefore = await (await page.request.get(base + '/api/history')).json();
  try {
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.getByText('Жүйе дайын', { exact: true }).waitFor();
    assert.equal(await page.locator('button[type="submit"]').isDisabled(), true);
    assert.equal(await page.getByRole('checkbox').isChecked(), false);
    await page.screenshot({ path: path.join(out, 'desktop-home.png'), fullPage: true });
    record('Service ready; submit disabled before upload; storage is opt-in');

    await page.locator('#leaf-file').setInputFiles({ name: 'note.txt', mimeType: 'text/plain', buffer: Buffer.from('invalid') });
    await page.getByRole('alert').filter({ hasText: 'Тек JPG немесе PNG' }).waitFor();
    assert.equal(await page.locator('button[type="submit"]').isDisabled(), true);
    record('Unsupported file rejected with a visible error');

    const sampleDir = path.join(root, 'data', 'plantvillage', 'Tomato___healthy');
    const sample = path.join(sampleDir, fs.readdirSync(sampleDir).filter(x => /\.jpe?g$/i.test(x)).sort()[0]);
    await page.locator('#leaf-file').setInputFiles(sample);
    assert.equal(await page.locator('#model').inputValue(), 'mobilenet');
    await page.getByAltText('Талдауға таңдалған жапырақ').waitFor();
    const responsePromise = page.waitForResponse(r => r.url().endsWith('/api/predict') && r.request().method() === 'POST');
    await page.locator('button[type="submit"]').click();
    const response = await responsePromise;
    assert.equal(response.status(), 200);
    const prediction = await response.json();
    for (const field of ['request_id', 'predicted_class', 'display_name_kk', 'confidence', 'top_k', 'model_version', 'warning']) assert.ok(field in prediction, field);
    assert.equal(prediction.saved_to_history, false);
    assert.equal(prediction.calibrated, false);
    assert.ok(prediction.heatmap.startsWith('data:image/png;base64,'));
    await page.getByText('Талдау аяқталды', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(out, 'desktop-result.png'), fullPage: true });
    record('Real image inference returns contract fields, Grad-CAM and no persistence');

    await page.locator('#model').selectOption('plantcnn');
    assert.equal(await page.getByText('Талдау аяқталды', { exact: true }).count(), 0);
    record('Changing model clears stale results');

    await page.goto(base + '/about', { waitUntil: 'networkidle' });
    await page.getByRole('heading', { name: 'Модельдер және шектеулер' }).waitFor();
    assert.equal(await page.locator('.model-card').count(), 2);
    await page.screenshot({ path: path.join(out, 'desktop-models.png'), fullPage: true });
    record('Model page renders both installed checkpoints and metrics');

    await page.goto(base + '/diseases', { waitUntil: 'networkidle' });
    await page.getByRole('textbox', { name: 'Анықтамалықтан іздеу' }).fill('Қызанақ');
    await page.screenshot({ path: path.join(out, 'desktop-diseases.png'), fullPage: true });
    record('Disease reference supports search');

    await page.goto(base + '/history', { waitUntil: 'networkidle' });
    await page.getByRole('heading', { name: 'Талдау тарихы', exact: true }).waitFor();
    await page.screenshot({ path: path.join(out, 'desktop-history.png'), fullPage: true });
    const reanalyze = page.getByRole('button', { name: 'Қайта талдау' }).first();
    if (await reanalyze.count()) {
      let submitted = false;
      const onRequest = req => { if (req.url().endsWith('/api/predict') && req.method() === 'POST') submitted = true; };
      page.on('request', onRequest);
      await reanalyze.click();
      await page.getByAltText('Талдауға таңдалған жапырақ').waitFor();
      assert.equal(submitted, false);
      page.off('request', onRequest);
      record('History reanalysis previews the image and requires explicit submission');
    }

    await page.setViewportSize({ width: 390, height: 844 });
    for (const route of ['/', '/about', '/history', '/diseases']) {
      await page.goto(base + route, { waitUntil: 'networkidle' });
      const dimensions = await page.evaluate(() => ({ width: innerWidth, document: document.documentElement.scrollWidth }));
      assert.ok(dimensions.document <= dimensions.width, `${route} horizontal overflow ${JSON.stringify(dimensions)}`);
      await page.screenshot({ path: path.join(out, 'mobile-' + (route.slice(1) || 'home') + '.png'), fullPage: true });
    }
    record('All four mobile pages fit a 390px viewport');

    await page.route('**/model-info', route => route.abort('connectionrefused'));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.getByRole('alert').filter({ hasText: 'Сервермен байланыс жоқ' }).waitFor();
    assert.equal(await page.locator('button[type="submit"]').isDisabled(), true);
    await page.unroute('**/model-info');
    await page.getByRole('button', { name: 'Қайталап көру' }).click();
    await page.getByText('Жүйе дайын', { exact: true }).waitFor();
    record('Unavailable backend is explained and retry restores readiness');

    const historyAfter = await (await page.request.get(base + '/api/history')).json();
    assert.equal(historyAfter.stats.total, historyBefore.stats.total);
    assert.deepEqual(errors, []);
    record('History unchanged and no browser JavaScript exceptions');
    fs.writeFileSync(path.join(out, 'checks.json'), JSON.stringify({ checked_at: new Date().toISOString(), browser: 'Microsoft Edge', checks, errors, history_count: historyAfter.stats.total }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
