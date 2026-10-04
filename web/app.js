"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  config: null,
  selection: null,
  job: null,
  jobs: [],
  picker: null,
  poll: null,
  connected: true,
  viewRequest: 0,
  historyRequest: 0,
  deleteTarget: null,
  deleting: false,
};
const statusLabels = {
  queued: "대기 중",
  running: "실행 중",
  completed: "완료",
  failed: "오류 확인",
  interrupted: "중단됨",
};
const phaseLabels = {
  queued: "작업 준비 중",
  analyzing: "원본 영상 분석 중",
  encoding: "영상 인코딩 중",
  validating: "영상 구조 확인 중",
  measuring: "PSNR · SSIM 검사 중",
  completed: "작업 완료",
  failed: "작업 종료 · 오류 확인",
  interrupted: "작업 중단",
};
let toastTimer;

function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );
}
function icon(name) {
  return `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
}
function bytes(value, precision = 1) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(
    Math.floor(Math.log(Math.abs(value)) / Math.log(1024)),
    units.length - 1,
  );
  return `${(value / 1024 ** index).toLocaleString("ko-KR", { maximumFractionDigits: index ? precision : 0 })} ${units[index]}`;
}
function metric(value, precision) {
  if (value === "inf") return "∞";
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(precision)
    : "—";
}
function notify(message, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  toastTimer = setTimeout(() => {
    $("toast").hidden = true;
  }, 5000);
}
async function api(path, data, method = "POST") {
  const response = await fetch(
    path,
    data === undefined
      ? { cache: "no-store" }
      : {
          method,
          headers: {
            "Content-Type": "application/json",
            "X-CSRF-Token": state.config.token,
          },
          body: JSON.stringify(data),
        },
  );
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.error || `요청 오류 (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}
function setConnection(connected) {
  state.connected = connected;
  $("server-dot").style.background =
    connected && state.config?.ffmpeg_ready ? "#6b9b7b" : "#d98a6c";
  $("server-status").textContent = !connected
    ? "서버 연결 확인 필요"
    : state.config?.ffmpeg_ready
      ? "FFmpeg 준비됨"
      : "FFmpeg 설치 필요";
  updateStartButton();
}
function updateStartButton() {
  const active =
    state.jobs.some((job) => ["running", "queued"].includes(job.status)) ||
    ["running", "queued"].includes(state.job?.status);
  $("start-job").disabled =
    !state.selection?.files.length ||
    !state.config?.ffmpeg_ready ||
    !state.connected ||
    active;
  $("start-job").innerHTML =
    `${active ? "작업 실행 중" : "트랜스코딩 시작"}${icon("arrow")}`;
}
function renderSelection() {
  const selection = state.selection;
  $("source-empty").hidden = Boolean(selection);
  $("selected-source").hidden = !selection;
  $("selection-count").textContent = selection
    ? `${selection.files.length.toLocaleString()}개 영상`
    : "선택 없음";
  if (selection) {
    $("selected-label").textContent = selection.label;
    $("selected-meta").textContent =
      `${selection.files.length.toLocaleString()}개 영상 · ${bytes(selection.total_size)} · MP4`;
    $("selected-files").innerHTML =
      selection.files
        .slice(0, 200)
        .map(
          (file) =>
            `<div class="selected-file">${icon("video")}<div class="file-text"><strong>${escapeHtml(file.name)}</strong><small title="${escapeHtml(file.path)}">${escapeHtml(file.path)}</small></div><span class="file-size">${bytes(file.size)}</span></div>`,
        )
        .join("") +
      (selection.files.length > 200
        ? `<div class="loading-text">외 ${(selection.files.length - 200).toLocaleString()}개 영상 · 선택한 전체 파일을 처리합니다.</div>`
        : "");
  }
  updateStartButton();
}
function applyPreset(name) {
  document
    .querySelectorAll(".preset")
    .forEach((button) =>
      button.classList.toggle("active", button.dataset.preset === name),
    );
  if (name !== "custom") {
    $("psnr").value = name === "high" ? 42 : 40;
    $("ssim").value = name === "high" ? 0.95 : 0.93;
  }
}

async function openPicker(mode) {
  if (!state.config) {
    notify("서버에 연결 중입니다. 잠시 후 다시 시도해 주세요.", true);
    return;
  }
  state.picker = {
    mode,
    path: state.selection?.folder || state.config.initial_path,
    parent: null,
    selected: new Map(),
    entries: [],
    request: 0,
  };
  $("picker-title").textContent =
    mode === "folder" ? "영상 폴더 선택" : "영상 파일 선택";
  $("confirm-selection").innerHTML =
    `${mode === "folder" ? "이 폴더 선택" : "선택 완료"}${icon("check")}`;
  $("recursive").checked = false;
  document.querySelector(".recursive-option").hidden = mode !== "folder";
  $("locations").innerHTML = state.config.locations
    .map(
      (location, index) =>
        `<option value="${index}">${escapeHtml(location.name)}</option>`,
    )
    .join("");
  $("file-dialog").showModal();
  await browse(state.picker.path);
}
async function browse(path) {
  const picker = state.picker;
  if (!picker) return;
  const request = ++picker.request;
  $("browse-message").classList.remove("error");
  $("browse-message").textContent = "폴더를 불러오는 중…";
  $("browse-entries").innerHTML =
    '<p class="loading-text">선택 가능한 폴더와 영상을 찾고 있습니다.</p>';
  $("confirm-selection").disabled = true;
  try {
    const result = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    if (picker !== state.picker || request !== picker.request) return;
    picker.path = result.path;
    picker.parent = result.parent;
    picker.entries = result.entries;
    $("browse-path").value = result.path;
    $("parent-folder").disabled = !result.parent;
    $("browse-entries").innerHTML =
      result.entries
        .map((entry, index) =>
          entry.type === "folder"
            ? `<button class="browse-row folder" data-entry="${index}">${icon("folder")}<strong>${escapeHtml(entry.name)}</strong><span class="entry-tail">폴더</span>${icon("chevron")}</button>`
            : `<label class="browse-row ${picker.selected.has(entry.path) ? "selected" : ""}"><input type="checkbox" data-entry="${index}" ${picker.selected.has(entry.path) ? "checked" : ""} ${picker.mode === "folder" ? "disabled" : ""}>${icon("video")}<strong>${escapeHtml(entry.name)}</strong><span class="entry-tail">${bytes(entry.size)}</span></label>`,
        )
        .join("") ||
      '<p class="loading-text">이 폴더에는 선택할 수 있는 MP4 영상이 없습니다.</p>';
    $("browse-message").textContent = result.truncated
      ? "항목이 많아 앞 2,000개만 표시합니다. 경로 입력으로 다른 폴더로 이동할 수 있습니다."
      : modeHint();
    updatePickerCount();
  } catch (error) {
    if (picker !== state.picker || request !== picker.request) return;
    $("browse-entries").innerHTML =
      '<p class="loading-text">폴더를 열 수 없습니다. 다른 위치나 경로를 선택해 주세요.</p>';
    $("browse-message").classList.add("error");
    $("browse-message").textContent = error.message;
  }
}
function modeHint() {
  return state.picker.mode === "folder"
    ? "폴더를 열어 이동한 뒤 ‘이 폴더 선택’을 눌러 주세요."
    : "폴더는 클릭해서 열고, 영상은 체크해서 선택하세요. 여러 폴더의 파일도 함께 선택할 수 있습니다.";
}
function updatePickerCount() {
  if (!state.picker) return;
  const count = state.picker.selected.size;
  $("picker-selection-count").textContent =
    state.picker.mode === "folder"
      ? "현재 폴더의 MP4 영상"
      : count
        ? `${count.toLocaleString()}개 파일 선택`
        : "선택한 파일 없음";
  $("confirm-selection").disabled = state.picker.mode === "files" && !count;
}
async function confirmSelection() {
  const picker = state.picker;
  const paths =
    picker.mode === "folder" ? [picker.path] : [...picker.selected.keys()];
  $("confirm-selection").disabled = true;
  $("browse-message").textContent = "선택한 영상을 확인하는 중…";
  try {
    const result = await api("/api/selection", {
      paths,
      recursive: picker.mode === "folder" && $("recursive").checked,
    });
    state.selection = {
      ...result,
      folder: picker.mode === "folder" ? picker.path : null,
      label:
        picker.mode === "folder"
          ? picker.path
          : result.files[0].name +
            (result.files.length > 1 ? ` 외 ${result.files.length - 1}개` : ""),
    };
    renderSelection();
    $("file-dialog").close();
    state.picker = null;
    notify(`${result.files.length.toLocaleString()}개 영상을 선택했습니다.`);
  } catch (error) {
    $("browse-message").classList.add("error");
    $("browse-message").textContent = error.message;
    updatePickerCount();
  }
}

async function loadHistory() {
  const request = ++state.historyRequest;
  try {
    const jobs = (await api("/api/jobs")).jobs;
    if (request !== state.historyRequest) return state.jobs;
    state.jobs = jobs;
    if (state.job && !jobs.some((job) => job.id === state.job.id))
      clearViewedJob();
    renderHistory();
    setConnection(true);
    return state.jobs;
  } catch (error) {
    if (request !== state.historyRequest) return state.jobs;
    setConnection(false);
    return [];
  }
}
function renderHistory() {
  $("history-count").textContent = state.jobs.length;
  $("history-select").innerHTML =
    state.jobs
      .map(
        (job) =>
          `<option value="${job.id}">${escapeHtml(job.label)} · ${statusLabels[job.status]}</option>`,
      )
      .join("") || '<option value="">아직 실행 기록이 없습니다</option>';
  $("history-select").disabled = !state.jobs.length;
  if (state.job) $("history-select").value = state.job.id;
  $("history-list").innerHTML =
    state.jobs
      .slice(0, 30)
      .map((job) => {
        const date = new Date(job.created_at).toLocaleString("ko-KR", {
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        });
        return `<button class="history-item ${state.job?.id === job.id ? "selected" : ""}" data-job="${job.id}"><strong title="${escapeHtml(job.label)}">${escapeHtml(job.label)}</strong><small><span>${escapeHtml(date)}</span><span>${statusLabels[job.status] || job.status}</span></small></button>`;
      })
      .join("") ||
    '<p class="sidebar-empty">첫 작업을 시작해 보세요.<br>실행 기록이 여기에 남습니다.</p>';
  updateStartButton();
  $("delete-history").disabled =
    state.deleting ||
    !state.job ||
    ["running", "queued"].includes(state.job.status);
  $("clear-history").disabled =
    state.deleting ||
    !state.jobs.some((job) => !["running", "queued"].includes(job.status));
}
function clearViewedJob() {
  ++state.viewRequest;
  clearTimeout(state.poll);
  state.job = null;
  try {
    localStorage.removeItem("frame-studio-job");
  } catch (_) {
    /* Optional persistence. */
  }
  renderJob();
}
function openDeleteDialog(all = false) {
  if (
    state.deleting ||
    (!all && (!state.job || ["running", "queued"].includes(state.job.status)))
  )
    return;
  const count = state.jobs.filter(
    (job) => !["running", "queued"].includes(job.status),
  ).length;
  if (all && !count) return;
  state.deleteTarget = all ? { all: true } : { id: state.job.id };
  $("delete-title").textContent = all
    ? "완료 기록 전체 삭제"
    : "선택 기록 삭제";
  $("delete-description").textContent = all
    ? `삭제 시점에 완료·실패·중단된 모든 기록을 삭제합니다. 현재 ${count}개이며, 실행 중인 작업은 보존됩니다.`
    : `‘${state.job.label}’의 실행 기록을 삭제합니다.`;
  $("delete-location").textContent =
    state.config.history_directory + (all ? "" : "/" + state.job.id);
  $("delete-error").hidden = true;
  $("delete-dialog").showModal();
}
async function deleteHistory() {
  if (!state.deleteTarget || state.deleting) return;
  const target = state.deleteTarget;
  state.deleting = true;
  $("confirm-delete").disabled = true;
  $("cancel-delete").disabled = true;
  $("close-delete").disabled = true;
  renderHistory();
  try {
    const result = await api(
      target.all ? "/api/jobs" : `/api/jobs/${target.id}`,
      {},
      "DELETE",
    );
    ++state.historyRequest;
    state.jobs = state.jobs.filter(
      (job) => !result.deleted_ids.includes(job.id),
    );
    if (result.deleted_ids.includes(state.job?.id)) clearViewedJob();
    renderHistory();
    $("delete-dialog").close();
    state.deleteTarget = null;
    await loadHistory();
    if (!state.job && state.jobs.length) {
      const next =
        state.jobs.find((job) => ["running", "queued"].includes(job.status)) ||
        state.jobs[0];
      await selectJob(next.id);
    }
    notify(
      `${result.deleted_ids.length}개 기록을 삭제했습니다. 결과 파일은 보존되었습니다.`,
    );
  } catch (error) {
    $("delete-error").textContent = error.message;
    $("delete-error").hidden = false;
    await loadHistory();
  } finally {
    state.deleting = false;
    $("confirm-delete").disabled = false;
    $("cancel-delete").disabled = false;
    $("close-delete").disabled = false;
    renderHistory();
  }
}
async function selectJob(id) {
  const request = ++state.viewRequest;
  clearTimeout(state.poll);
  try {
    const job = await api(`/api/jobs/${id}`);
    if (request !== state.viewRequest) return;
    state.job = job;
    try {
      localStorage.setItem("frame-studio-job", id);
    } catch (_) {
      /* Private browser storage may be unavailable. */
    }
    renderJob();
    renderHistory();
    setConnection(true);
    if (["running", "queued"].includes(job.status))
      state.poll = setTimeout(() => selectJob(id), 1200);
    else await loadHistory();
  } catch (error) {
    if (request !== state.viewRequest) return;
    if (error.status === 404) {
      if (state.job?.id === id) clearViewedJob();
      await loadHistory();
      notify("삭제되었거나 존재하지 않는 실행 기록입니다.", true);
      return;
    }
    setConnection(false);
    if (state.job?.id === id)
      state.poll = setTimeout(() => selectJob(id), 3000);
    else notify(error.message, true);
  }
}
function renderJob() {
  const job = state.job;
  if (!job) {
    document.querySelector(".monitor").classList.remove("running", "has-job");
    $("monitor-state").textContent = "준비";
    $("monitor-id").textContent = "YOUR NEXT FRAME";
    $("monitor-title").textContent = "다음 장면을 준비하세요.";
    $("monitor-description").textContent =
      "작업을 시작하면 진행 상황과 화질 검사 결과를 확인할 수 있어요.";
    $("monitor-orbit").hidden = false;
    $("live-progress").hidden = true;
    ["analyzing", "encoding", "measuring"].forEach((stage) =>
      $("stage-" + stage).classList.remove("active", "done"),
    );
    [
      "live-psnr",
      "live-ssim",
      "summary-accepted",
      "summary-unchanged",
      "summary-saved",
      "summary-percent",
    ].forEach((id) => ($(id).textContent = "—"));
    $("results-caption").textContent =
      "작업이 끝나면 영상별 화질과 압축 결과를 확인할 수 있습니다.";
    $("results-empty").hidden = false;
    $("results-empty").innerHTML =
      `${icon("video")}<strong>아직 실행 결과가 없습니다</strong><span>영상을 선택하고 최적화를 시작해 보세요.</span>`;
    $("results-body").innerHTML = "";
    $("log-content").textContent = "";
    [
      "results-table-wrap",
      "report-actions",
      "job-error",
      "execution-log",
    ].forEach((id) => ($(id).hidden = true));
    ["csv", "json"].forEach((extension) =>
      $("download-" + extension).removeAttribute("href"),
    );
    updateStartButton();
    return;
  }
  const active = ["running", "queued"].includes(job.status);
  const success = job.status === "completed";
  document.querySelector(".monitor").classList.toggle("running", active);
  document.querySelector(".monitor").classList.add("has-job");
  $("monitor-state").textContent = statusLabels[job.status] || job.status;
  $("monitor-id").textContent = `JOB / ${job.id.slice(0, 8).toUpperCase()}`;
  $("monitor-title").textContent = active
    ? "더 가벼운 영상을 만드는 중."
    : success
      ? "다음 장면이 준비됐습니다."
      : "작업 결과를 확인해 주세요.";
  $("monitor-description").textContent = active
    ? `${job.total}개 영상의 화질과 용량을 확인합니다.`
    : `${job.summary.accepted}개 채택 · ${job.summary.unchanged}개 원본 유지 · ${job.summary.failed}개 실패`;
  $("monitor-orbit").hidden = true;
  $("live-progress").hidden = false;
  $("progress-label").textContent = phaseLabels[job.phase] || "작업 진행 중";
  $("progress-count").textContent = `${job.completed} / ${job.total}`;
  $("progress-fill").style.width =
    `${job.total ? (100 * job.completed) / job.total : 0}%`;
  $("current-file").textContent = active
    ? job.current_file || "작업을 준비하고 있습니다"
    : job.label;
  $("current-file").title = job.current_path || job.label;
  $("attempt-label").textContent =
    active && job.attempt
      ? `시도 ${job.attempt}/${job.max_attempts} · ${job.bitrate ? (job.bitrate / 1e6).toFixed(2) + " Mbps" : ""} · ${job.encoder === "cuda" ? "GPU" : "CPU"}`
      : active
        ? "영상별 완료 수를 표시합니다"
        : "원본 영상은 그대로 보존되었습니다";
  ["analyzing", "encoding", "measuring"].forEach((stage, index) => {
    const phases = { analyzing: 0, encoding: 1, validating: 2, measuring: 2 };
    $("stage-" + stage).classList.toggle(
      "active",
      active && phases[job.phase] === index,
    );
    $("stage-" + stage).classList.toggle(
      "done",
      success || (active && phases[job.phase] > index),
    );
  });
  const metrics =
    job.metrics ||
    (job.records.length ? job.records[job.records.length - 1] : {});
  $("live-psnr").textContent = metric(metrics.psnr_avg, 2);
  $("live-ssim").textContent = metric(metrics.ssim_all, 4);
  const summary = job.summary;
  $("summary-accepted").innerHTML = `${summary.accepted}<small>개</small>`;
  $("summary-unchanged").innerHTML = `${summary.unchanged}<small>개</small>`;
  $("summary-saved").textContent = bytes(summary.saved_size);
  $("summary-percent").innerHTML =
    `${summary.saved_percent.toFixed(1)}<small>%</small>`;
  $("results-caption").textContent =
    `${job.total}개 영상 · PSNR ${job.settings.psnr} dB / SSIM ${job.settings.ssim} 이상 · ${statusLabels[job.status]}`;
  $("results-empty").hidden = job.records.length > 0;
  $("results-table-wrap").hidden = !job.records.length;
  if (!job.records.length && active)
    $("results-empty").innerHTML =
      `${icon("video")}<strong>첫 영상의 결과를 기다리고 있습니다</strong><span>분석과 화질 검사가 끝나면 결과가 표시됩니다.</span>`;
  else if (!job.records.length)
    $("results-empty").innerHTML =
      `${icon("video")}<strong>저장된 결과가 없습니다</strong><span>실행 로그와 오류 메시지를 확인해 주세요.</span>`;
  $("results-body").innerHTML = job.records
    .map((record) => {
      const accepted = record.status === "accepted";
      const reduction =
        accepted && Number.isFinite(record.ratio)
          ? `${((1 - record.ratio) * 100).toFixed(1)}%`
          : "—";
      const label =
        accepted && !record.output_path
          ? "검증 통과 · 저장 대기"
          : { accepted: "채택", unchanged: "원본 유지", failed: "실패" }[
              record.status
            ] || "확인 필요";
      const bitrate = (value) =>
        typeof value === "number" ? (value / 1e6).toFixed(2) + " Mbps" : "—";
      return `<tr><td><strong>${escapeHtml(record.file_name)}</strong><small title="${escapeHtml(record.source_path)}">${escapeHtml(record.source_path)}</small>${record.reason ? `<small>${escapeHtml(record.reason)}</small>` : ""}</td><td><span class="result-badge ${escapeHtml(record.status)}">${label}</span></td><td class="metric-value">${metric(record.psnr_avg, 2)} dB<small>Y ${metric(record.psnr_y, 2)} dB · ${record.attempts || 0}회 시도</small><small>SSIM ${metric(record.ssim_all, 4)} / Y ${metric(record.ssim_y, 4)}</small></td><td class="metric-value">${bytes(record.orig_file_size)}<small>→ ${accepted ? bytes(record.trans_file_size) : "원본 유지"}</small><small>${bitrate(record.orig_video_bitrate)}${accepted ? " → " + bitrate(record.trans_video_bitrate) : ""}</small></td><td class="metric-value">${reduction}</td><td>${record.download_url ? `<a class="download-link" href="${escapeHtml(record.download_url)}" aria-label="${escapeHtml(record.file_name)} 다운로드">${icon("download")}</a>` : ""}</td></tr>`;
    })
    .join("");
  $("report-actions").hidden = !job.reports?.csv && !job.reports?.json;
  ["csv", "json"].forEach((extension) => {
    $("download-" + extension).hidden = !job.reports?.[extension];
    if (job.reports?.[extension])
      $("download-" + extension).href = job.reports[extension];
  });
  $("job-error").hidden = !job.error;
  $("job-error").textContent = job.error || "";
  $("execution-log").hidden = !job.logs.length;
  $("log-count").textContent = `${job.logs.length} lines`;
  if ($("log-content").textContent !== job.logs.join("\n")) {
    const wasBottom =
      $("log-content").scrollHeight -
        $("log-content").scrollTop -
        $("log-content").clientHeight <
      30;
    $("log-content").textContent = job.logs.join("\n");
    if (wasBottom) $("log-content").scrollTop = $("log-content").scrollHeight;
  }
  updateElapsed();
  updateStartButton();
}
function updateElapsed() {
  if (!state.job?.started_at) return;
  const seconds = Math.max(
    0,
    Math.floor(
      ((state.job.finished_at
        ? new Date(state.job.finished_at).getTime()
        : Date.now()) -
        new Date(state.job.started_at).getTime()) /
        1000,
    ),
  );
  $("elapsed-time").textContent = `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}
async function startJob() {
  if (!state.selection) return;
  if (!$("psnr").reportValidity() || !$("ssim").reportValidity()) return;
  const ratios = $("ratios")
    .value.split(/[,\s]+/)
    .filter(Boolean)
    .map(Number);
  if (
    !ratios.length ||
    ratios.some(
      (ratio, index) =>
        !Number.isFinite(ratio) ||
        ratio <= 0 ||
        ratio >= 1 ||
        (index > 0 && ratio <= ratios[index - 1]),
    )
  ) {
    notify(
      "비트레이트 비율은 0과 1 사이의 중복 없는 오름차순으로 입력해 주세요.",
      true,
    );
    $("ratios").closest("details").open = true;
    $("ratios").focus();
    return;
  }
  $("start-job").disabled = true;
  state.viewRequest += 1;
  clearTimeout(state.poll);
  try {
    const job = await api("/api/jobs", {
      paths: state.selection.files.map((file) => file.path),
      settings: {
        psnr: Number($("psnr").value),
        ssim: Number($("ssim").value),
        encoder: $("encoder").value,
        ratios,
      },
    });
    state.job = job;
    renderJob();
    await loadHistory();
    await selectJob(job.id);
    notify("트랜스코딩을 시작했습니다. 원본 영상은 보존됩니다.");
  } catch (error) {
    notify(error.message, true);
    updateStartButton();
  }
}

$("choose-files").addEventListener("click", () => openPicker("files"));
$("choose-folder").addEventListener("click", () => openPicker("folder"));
$("change-source").addEventListener("click", () =>
  openPicker(state.selection?.folder ? "folder" : "files"),
);
$("close-picker").addEventListener("click", () => {
  $("file-dialog").close();
  state.picker = null;
});
$("file-dialog").addEventListener("cancel", () => {
  state.picker = null;
});
$("path-form").addEventListener("submit", (event) => {
  event.preventDefault();
  browse($("browse-path").value);
});
$("locations").addEventListener("change", () =>
  browse(state.config.locations[Number($("locations").value)].path),
);
$("parent-folder").addEventListener("click", () => {
  if (state.picker?.parent) browse(state.picker.parent);
});
$("browse-entries").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-entry]");
  if (button) browse(state.picker.entries[Number(button.dataset.entry)].path);
});
$("browse-entries").addEventListener("change", (event) => {
  const entry = state.picker.entries[Number(event.target.dataset.entry)];
  if (event.target.checked) state.picker.selected.set(entry.path, entry);
  else state.picker.selected.delete(entry.path);
  event.target
    .closest(".browse-row")
    .classList.toggle("selected", event.target.checked);
  updatePickerCount();
});
$("confirm-selection").addEventListener("click", confirmSelection);
document
  .querySelectorAll(".preset")
  .forEach((button) =>
    button.addEventListener("click", () => applyPreset(button.dataset.preset)),
  );
[$("psnr"), $("ssim")].forEach((input) =>
  input.addEventListener("input", () => applyPreset("custom")),
);
$("start-job").addEventListener("click", startJob);
$("history-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-job]");
  if (button) {
    await selectJob(button.dataset.job);
    $("results-section").scrollIntoView({ behavior: "smooth" });
  }
});
$("refresh-history").addEventListener("click", loadHistory);
$("delete-history").addEventListener("click", () => openDeleteDialog());
$("clear-history").addEventListener("click", () => openDeleteDialog(true));
$("confirm-delete").addEventListener("click", deleteHistory);
[$("cancel-delete"), $("close-delete")].forEach((button) =>
  button.addEventListener("click", () => {
    $("delete-dialog").close();
    state.deleteTarget = null;
  }),
);
$("delete-dialog").addEventListener("cancel", (event) => {
  if (state.deleting) event.preventDefault();
  else state.deleteTarget = null;
});
$("history-select").addEventListener("change", () =>
  selectJob($("history-select").value),
);
$("workspace-nav").addEventListener("click", () => {
  document.querySelector("main").scrollIntoView({ behavior: "smooth" });
  $("workspace-nav").classList.add("active");
  $("history-nav").classList.remove("active");
});
$("history-nav").addEventListener("click", async () => {
  await loadHistory();
  $("results-section").scrollIntoView({ behavior: "smooth" });
  $("history-nav").classList.add("active");
  $("workspace-nav").classList.remove("active");
  if (!state.job && state.jobs.length) selectJob(state.jobs[0].id);
});
$("help-button").addEventListener("click", () => $("help-dialog").showModal());
$("close-help").addEventListener("click", () => $("help-dialog").close());
setInterval(updateElapsed, 1000);
setInterval(loadHistory, 10000);

async function initialize() {
  try {
    state.config = await api("/api/config");
    setConnection(true);
    const jobs = await loadHistory();
    let previous;
    try {
      previous = localStorage.getItem("frame-studio-job");
    } catch (_) {
      /* Optional persistence. */
    }
    const active = jobs.find((job) =>
      ["running", "queued"].includes(job.status),
    );
    const job = active || jobs.find((job) => job.id === previous) || jobs[0];
    if (job) await selectJob(job.id);
  } catch (error) {
    setConnection(false);
    notify(`서버에 연결할 수 없습니다: ${error.message}`, true);
  }
}
initialize();
