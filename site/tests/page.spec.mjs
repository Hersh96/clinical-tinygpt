import { test, after, before } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { serve } from "./server.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const root = fileURLToPath(new URL("../public", import.meta.url));
const PAGE = "/projects/clinical-tinygpt/index.html";
const CHAPTERS = ["start", "tokenization", "transformer", "training", "scaling", "kv-cache", "results"];
let browser, site, blocked;

before(async () => {
  browser = await chromium.launch();
  site = await serve(root);
  blocked = await serve(root, new Set(["/projects/clinical-tinygpt/data/scaling.json", "/projects/clinical-tinygpt/data/summary.json"]));
});
after(async () => { await browser.close(); site.server.close(); blocked.server.close(); });

async function open(base, opts = {}) {
  const page = await browser.newPage(opts);
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(base + PAGE);
  await page.waitForSelector("body[data-ready='1']", { timeout: 15000 });
  return { page, errors };
}

test("every chapter renders, with no script errors", async () => {
  const { page, errors } = await open(site.url);
  for (const id of CHAPTERS) assert.ok(await page.isVisible(`#${id} h2, #${id} h1`), id);
  assert.deepEqual(errors, []);
  assert.ok((await page.locator("canvas").count()) >= 4);
  await page.close();
});

test("tokenizer playground splits typed text into chips", async () => {
  const { page } = await open(site.url);
  await page.fill("#tok-input", "Metformin 500 mg");
  await page.waitForFunction(() => document.querySelectorAll("#tok-chips .chip").length >= 1);
  const text = await page.$$eval("#tok-chips .chip", (els) => els.map((e) => e.dataset.piece).join(""));
  assert.equal(text, "Metformin 500 mg");
  await page.selectOption("#tok-vocab", "4096");
  await page.waitForFunction(() => document.querySelector("#tok-count").textContent.length > 0);
  await page.close();
});

test("non-ASCII text survives the playground", async () => {
  const { page } = await open(site.url);
  await page.fill("#tok-input", "5 µg at 37 °C 🙂");
  await page.waitForFunction(() => document.querySelectorAll("#tok-chips .chip").length >= 1);
  assert.ok((await page.textContent("#tok-count")).match(/\d+ tokens/));
  await page.close();
});

test("no horizontal scroll at phone width", async () => {
  const { page } = await open(site.url, { viewport: { width: 375, height: 800 } });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  assert.ok(overflow <= 0, `page is ${overflow}px wider than the screen`);
  await page.close();
});

test("dark mode follows the system setting", async () => {
  const light = await open(site.url, { colorScheme: "light" });
  const dark = await open(site.url, { colorScheme: "dark" });
  const bg = (p) => p.evaluate(() => getComputedStyle(document.body).backgroundColor);
  assert.notEqual(await bg(light.page), await bg(dark.page));
  await light.page.close(); await dark.page.close();
});

test("a missing data file shows a message instead of a blank chart", async () => {
  const { page, errors } = await open(blocked.url);
  assert.match(await page.textContent("#scaling"), /Data not available yet/);
  assert.match(await page.textContent("#start"), /Data not available yet/);
  assert.ok(await page.isVisible("#chart-loss"));
  assert.deepEqual(errors, []);
  await page.close();
});

test("glossary terms open a definition", async () => {
  const { page } = await open(site.url);
  await page.click("button.term >> nth=0");
  assert.ok(await page.isVisible("#glossary-pop"));
  await page.close();
});
