// Optional browser check: start Chrome with --headless=new --remote-debugging-port=9222
// and a separate --user-data-dir, then run: node tests/browser_smoke.mjs
import { writeFile, mkdir, access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

const origin = process.env.FRAME_STUDIO_URL || "http://127.0.0.1:8765";
const debugOrigin = process.env.CHROME_DEBUG_URL || "http://127.0.0.1:9222";
const output = resolve(process.env.FRAME_STUDIO_ARTIFACT_DIR || ".web-data/browser-artifacts");
const fixture = resolve(process.env.FRAME_STUDIO_FIXTURE_DIR || ".web-data/ui-fixtures");
await mkdir(fixture, { recursive: true });
try {
  await access(resolve(fixture, "Studio sample.mp4"));
} catch {
  execFileSync(
    "ffmpeg",
    [
      "-nostdin",
      "-v",
      "error",
      "-f",
      "lavfi",
      "-i",
      "testsrc2=size=320x180:rate=24",
      "-t",
      "2",
      "-c:v",
      "libx264",
      "-qp",
      "0",
      "-preset",
      "ultrafast",
      resolve(fixture, "Studio sample.mp4"),
    ],
    { windowsHide: true, stdio: "pipe" },
  );
}
const targets = await (await fetch(`${debugOrigin}/json/list`)).json();
const target = targets.find(
  (entry) => entry.type === "page" && entry.url.startsWith(origin),
);
if (!target)
  throw new Error(
    "Open Frame Studio in the headless Chrome test profile first.",
  );
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((done, reject) => {
  socket.addEventListener("open", done, { once: true });
  socket.addEventListener("error", reject, { once: true });
});
let sequence = 0;
const pending = new Map();
const errors = [];
socket.addEventListener("message", (message) => {
  const result = JSON.parse(message.data);
  if (result.method === "Runtime.exceptionThrown")
    errors.push(result.params.exceptionDetails);
  if (!result.id) return;
  const callback = pending.get(result.id);
  if (!callback) return;
  pending.delete(result.id);
  result.error
    ? callback.reject(new Error(JSON.stringify(result.error)))
    : callback.resolve(result.result);
});
function send(method, params = {}) {
  return new Promise((resolvePromise, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`Timed out: ${method}`));
    }, 15000);
    pending.set(id, {
      resolve: (value) => {
        clearTimeout(timer);
        resolvePromise(value);
      },
      reject: (error) => {
        clearTimeout(timer);
        reject(error);
      },
    });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}
async function until(expression, timeout = 15000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const result = await evaluate(expression);
    if (result) return result;
    await new Promise((done) => setTimeout(done, 100));
  }
  throw new Error(`Condition not met: ${expression}`);
}
async function screenshot(name, fullPage = false) {
  await new Promise((done) => setTimeout(done, 400));
  const params = { format: "png", captureBeyondViewport: fullPage };
  if (fullPage) {
    const metrics = await send("Page.getLayoutMetrics");
    params.clip = {
      x: 0,
      y: 0,
      width: metrics.cssContentSize.width,
      height: metrics.cssContentSize.height,
      scale: 1,
    };
  }
  const image = await send("Page.captureScreenshot", params);
  await writeFile(resolve(output, name), Buffer.from(image.data, "base64"));
}

try {
  await mkdir(output, { recursive: true });
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1100,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send("Page.navigate", { url: origin });
  await until(
    "document.querySelector('#server-status')?.textContent === 'FFmpeg 준비됨'",
  );
  await screenshot("desktop-initial.png", true);
  if (await evaluate("state.job === null && !document.querySelector('#replace-originals').disabled"))
    throw new Error("Replace must be disabled without successful results.");
  await evaluate("document.querySelector('#choose-files').click()");
  await until(
    "document.querySelector('#file-dialog').open && document.querySelector('#browse-entries .loading-text') === null || document.querySelector('#browse-message').textContent.includes('선택')",
  );
  await evaluate(
    `document.querySelector('#browse-path').value = ${JSON.stringify(fixture)}; document.querySelector('#path-form').requestSubmit();`,
  );
  await until("document.querySelector('.browse-row input') !== null");
  await screenshot("file-picker.png");
  await evaluate("document.querySelector('.browse-row input').click()");
  await evaluate("document.querySelector('#confirm-selection').click()");
  await until(
    "!document.querySelector('#file-dialog').open && !document.querySelector('#start-job').disabled",
  );
  const previousId = await evaluate(
    "document.querySelector('#monitor-id').textContent",
  );
  await evaluate(
    "document.querySelector('#encoder').value = 'cpu'; document.querySelector('#start-job').click()",
  );
  await until(
    `document.querySelector('#monitor-id').textContent !== ${JSON.stringify(previousId)}`,
  );
  await until(
    "document.querySelector('#monitor-state').textContent === '완료'",
    60000,
  );
  await until(
    "document.querySelector('#download-csv').getAttribute('href') !== null",
  );
  if (
    !(await evaluate(
      "document.querySelector('#results-body .result-badge')?.textContent === '채택'",
    ))
  )
    throw new Error("Expected an accepted video.");
  await screenshot("desktop-result.png", true);
  if (await evaluate("document.querySelector('#replace-originals').disabled"))
    throw new Error("Replace must be enabled for a saved successful result.");
  await send("Page.reload");
  await until(
    "document.querySelector('#monitor-state')?.textContent === '완료' && document.querySelector('#history-count')?.textContent !== '0'",
  );
  const reportHref = await evaluate(
    "document.querySelector('#download-json').getAttribute('href')",
  );
  const reportResponse = await fetch(`${origin}${reportHref}`);
  const report = await reportResponse.json();
  if (!reportResponse.ok || report[0].status !== "accepted")
    throw new Error("Report download failed.");
  await evaluate("document.querySelector('#choose-folder').click()");
  await until(
    "document.querySelector('#file-dialog').open && !document.querySelector('#confirm-selection').disabled",
  );
  await evaluate(
    `document.querySelector('#browse-path').value = ${JSON.stringify(fixture)}; document.querySelector('#path-form').requestSubmit();`,
  );
  await until(
    `document.querySelector('#browse-path').value === ${JSON.stringify(fixture)} && document.querySelector('.browse-row input') !== null`,
  );
  await evaluate(
    "document.querySelector('#recursive').checked = true; document.querySelector('#confirm-selection').click()",
  );
  await until(
    "!document.querySelector('#file-dialog').open && !document.querySelector('#selected-source').hidden",
  );
  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await evaluate("window.scrollTo(0, 0)");
  await screenshot("mobile-result.png", true);
  const overflow = await evaluate(
    "document.documentElement.scrollWidth > document.documentElement.clientWidth + 1",
  );
  if (overflow) throw new Error("Mobile layout overflows the viewport.");
  // Only overwrite the generated fixture on an explicitly isolated test run.
  if (process.env.FRAME_STUDIO_CHECK_REPLACE === "1") {
    if (resolve(report[0].source_path) !== resolve(fixture, "Studio sample.mp4"))
      throw new Error("Replace target is not the generated sample video.");
    const jobId = reportHref.split("/")[3];
    const compressed = await readFile(report[0].output_path);
    await evaluate("document.querySelector('#replace-originals').click()");
    await until("document.querySelector('#results-body .result-badge')?.textContent === '원본 대체 완료'");
    await until("state.replacing === null && document.querySelector('#replace-originals').disabled");
    const original = await readFile(report[0].source_path);
    if (!original.equals(compressed)) throw new Error("Original was not replaced with the result video.");
    const groupReportDirectory = resolve(report[0].output_path, "..");
    for (const extension of ["json", "csv"]) {
      const copied = await readFile(resolve(fixture, `measured_data_${jobId}.${extension}`));
      const sourceReport = await readFile(resolve(groupReportDirectory, `measured_data.${extension}`));
      if (!copied.equals(sourceReport)) throw new Error(`${extension} report was not copied correctly.`);
    }
    await send("Page.reload");
    await until("document.querySelector('#results-body .result-badge')?.textContent === '원본 대체 완료' && document.querySelector('#replace-originals').disabled");
    await screenshot("mobile-after-replace.png", true);
  }
  // Enable deletion checks against an isolated test server/data directory.
  if (process.env.FRAME_STUDIO_CHECK_DELETE === "1") {
    const jobId = reportHref.split("/")[3];
    const selectedId = await evaluate(
      "document.querySelector('#history-select').value",
    );
    if (selectedId !== jobId)
      throw new Error("Deletion target changed unexpectedly.");
    await evaluate("document.querySelector('#clear-history').click()");
    await until("document.querySelector('#delete-dialog').open");
    await evaluate("document.querySelector('#cancel-delete').click()");
    await evaluate("document.querySelector('#delete-history').click()");
    await until("document.querySelector('#delete-dialog').open");
    await screenshot("mobile-delete-dialog.png");
    if (
      await evaluate(
        "document.documentElement.scrollWidth > document.documentElement.clientWidth + 1",
      )
    ) {
      throw new Error("Deletion dialog overflows the mobile viewport.");
    }
    await evaluate("document.querySelector('#cancel-delete').click()");
    if (!(await fetch(`${origin}/api/jobs/${jobId}`)).ok)
      throw new Error("Cancel removed history.");
    await evaluate("document.querySelector('#delete-history').click()");
    await until("document.querySelector('#delete-dialog').open");
    await evaluate("document.querySelector('#confirm-delete').click()");
    await until(
      `!document.querySelector('#delete-dialog').open && !Array.from(document.querySelector('#history-select').options).some(option => option.value === ${JSON.stringify(jobId)})`,
    );
    if ((await fetch(`${origin}/api/jobs/${jobId}`)).status !== 404)
      throw new Error("Deleted history is still accessible.");
    await access(report[0].output_path);
    await send("Page.reload");
    await until(
      "document.querySelector('#server-status')?.textContent === 'FFmpeg 준비됨'",
    );
    if (
      await evaluate(
        `Array.from(document.querySelector('#history-select').options).some(option => option.value === ${JSON.stringify(jobId)})`,
      )
    ) {
      throw new Error("Deleted history returned after reload.");
    }
    await screenshot("mobile-after-delete.png", true);
  }
  if (errors.length)
    throw new Error(`Browser runtime errors: ${JSON.stringify(errors)}`);
  console.log(
    JSON.stringify(
      { passed: true, report: report[0], screenshots: output },
      null,
      2,
    ),
  );
} finally {
  socket.close();
}
