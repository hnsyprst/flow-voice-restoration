"use strict";

// What the API's formatter and validator expect (`AUDIO_FORMATTER_44KHZ_MONO`, `AUDIO_VALIDATOR_44KHZ_MONO_5MIN`)
const SAMPLE_RATE = 44100;
const MIN_SECONDS = 1;
const MAX_SECONDS = 300;

const EASE_OUT = "cubic-bezier(0.22, 1, 0.36, 1)";
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

const background = new AsciiBackground(document.getElementById("background"));
background.start();

const glass = document.getElementById("glass");
const dropzone = document.getElementById("dropzone");
const idleView = document.getElementById("idle");
const recordingView = document.getElementById("recording");
const recordingTime = document.getElementById("recording-time");
const errorText = document.getElementById("error");
const recordButton = document.getElementById("record");
const stopButton = document.getElementById("stop");
const uploadButton = document.getElementById("upload");
const fileInput = document.getElementById("file");
const originalRow = document.getElementById("original-row");
const enhancedRow = document.getElementById("enhanced-row");
const enhanceButton = document.getElementById("enhance");
const progressBar = document.getElementById("progress");
const progressFill = document.getElementById("progress-fill");
const downloadLink = document.getElementById("download");

// Liquid glass refraction in Chromium; a frosted blur elsewhere
const liquidOptions = { scale: -170, chroma: 1.5, border: 0.02, mapBlur: 12, blur: 1, saturate: 1 };
let liquid = liquidGlass(glass, liquidOptions);

let audioContext = null;
let input = null;
let busy = false;

// ---------------------------------------------------------------------------------------------------------------------
// Audio

/**
 * Decode any audio the browser can read to mono samples at `SAMPLE_RATE`.
 *
 * @param {ArrayBuffer} data Encoded audio.
 * @returns {Promise<Float32Array>} The samples.
 */
async function decodeToMono(data) {
  audioContext ??= new AudioContext();
  const decoded = await audioContext.decodeAudioData(data);
  // A mono destination downmixes the source (to the mean of its channels) as it resamples it
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE)), SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  return (await offline.startRendering()).getChannelData(0);
}

/**
 * Encode mono samples at `SAMPLE_RATE` as a 16-bit PCM wav.
 *
 * @param {Float32Array} samples The samples.
 * @returns {Blob} The wav.
 */
function encodeWav(samples) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const write = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function formatTime(seconds) {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/**
 * A row's play button, waveform and time readout, playing one sample.
 */
class Player {
  static all = [];

  constructor(row) {
    this.button = row.querySelector(".play");
    this.canvas = row.querySelector(".wave");
    this.time = row.querySelector(".time");
    this.audio = new Audio();
    this.peaks = null;
    this.url = null;
    Player.all.push(this);

    this.button.addEventListener("click", () => this.toggle());
    this.canvas.addEventListener("click", (event) => this.seek(event));
    this.audio.addEventListener("play", () => {
      this.button.classList.add("playing");
      this.animate();
    });
    this.audio.addEventListener("pause", () => this.button.classList.remove("playing"));
    this.audio.addEventListener("ended", () => this.draw());
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }

  /**
   * @param {Float32Array} samples Mono samples at `SAMPLE_RATE`, for the waveform.
   * @param {Blob} blob The audio to play.
   */
  load(samples, blob) {
    this.unload();
    this.samples = samples;
    this.duration = samples.length / SAMPLE_RATE;
    this.url = URL.createObjectURL(blob);
    this.audio.src = this.url;
    this.button.disabled = false;
    this.draw();
  }

  unload() {
    this.audio.pause();
    this.audio.removeAttribute("src");
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
    this.samples = null;
    this.button.disabled = true;
  }

  toggle() {
    if (!this.audio.paused) return this.audio.pause();
    Player.all.forEach((player) => player !== this && player.audio.pause());
    this.audio.play();
  }

  seek(event) {
    if (!this.samples) return;
    const bounds = this.canvas.getBoundingClientRect();
    this.audio.currentTime = ((event.clientX - bounds.left) / bounds.width) * this.duration;
    this.draw();
  }

  animate() {
    this.draw();
    if (!this.audio.paused) requestAnimationFrame(() => this.animate());
  }

  draw() {
    if (!this.samples) return;
    const ratio = devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width) return;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    const context = this.canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    // One bar per 3 px: the peak of its slice of the samples, normalised to the loudest bar
    const bars = Math.max(1, Math.floor(width / 3));
    if (this.peaks?.length !== bars || this.peaksFor !== this.samples) {
      const step = this.samples.length / bars;
      this.peaks = new Float32Array(bars);
      for (let i = 0; i < bars; i++) {
        let peak = 0;
        for (let j = Math.floor(i * step), end = Math.floor((i + 1) * step); j < end; j++) {
          peak = Math.max(peak, Math.abs(this.samples[j]));
        }
        this.peaks[i] = peak;
      }
      const loudest = Math.max(...this.peaks) || 1;
      this.peaks = this.peaks.map((peak) => peak / loudest);
      this.peaksFor = this.samples;
    }

    const played = this.audio.currentTime / this.duration;
    context.clearRect(0, 0, width, height);
    this.peaks.forEach((peak, i) => {
      const barHeight = Math.max(1, peak * height * 0.9);
      context.fillStyle = i / bars < played ? "#f2f2f2" : "rgb(242 242 242 / 25%)";
      context.fillRect(i * 3, (height - barHeight) / 2, 2, barHeight);
    });
    this.time.textContent = `${formatTime(this.audio.currentTime)} / ${formatTime(this.duration)}`;
  }
}

const original = new Player(originalRow);
const enhanced = new Player(enhancedRow);

// ---------------------------------------------------------------------------------------------------------------------
// Motion

/**
 * Animate an element's height across a change to its contents.
 *
 * @param {HTMLElement} element The element whose height changes.
 * @param {() => void} change Changes the contents.
 */
function animateHeight(element, change) {
  const from = element.getBoundingClientRect().height;
  change();
  const to = element.getBoundingClientRect().height;
  liquid.refresh();
  if (reducedMotion.matches || from === to) return;
  const animation = element.animate([{ height: `${from}px` }, { height: `${to}px` }], { duration: 450, easing: EASE_OUT });

  // The glass's displacement map covers its old size, so redraw it every frame while it grows or shrinks
  let animating = true;
  const stop = () => {
    animating = false;
    liquid.refresh();
  };
  animation.addEventListener("finish", stop);
  animation.addEventListener("cancel", stop);
  const refresh = () => {
    if (!animating) return;
    liquid.refresh();
    requestAnimationFrame(refresh);
  };
  refresh();
}

/**
 * Fade, slide and unblur an element in, as it's revealed.
 */
function enter(element, delay = 150) {
  if (reducedMotion.matches) return;
  element.animate(
    [
      { opacity: 0, transform: "translateY(-8px)", filter: "blur(4px)" },
      { opacity: 1, transform: "none", filter: "blur(0)" },
    ],
    { duration: 300, delay, easing: "cubic-bezier(0.33, 1, 0.68, 1)", fill: "backwards" },
  );
}

/**
 * Swap one element for another, animating the panel's height.
 */
function swap(from, to) {
  animateHeight(glass, () => {
    from.hidden = true;
    to.hidden = false;
  });
  enter(to, 0);
}

function showError(message) {
  animateHeight(glass, () => {
    errorText.textContent = message ?? "";
    errorText.hidden = !message;
  });
}

function setBusy(value) {
  busy = value;
  for (const button of [recordButton, uploadButton, enhanceButton]) button.disabled = value;
}

// ---------------------------------------------------------------------------------------------------------------------
// Input

/**
 * Load a sample for enhancement, revealing the original row.
 *
 * @param {ArrayBuffer} data Encoded audio.
 * @param {string} name The file name, for the enhanced file's name.
 */
async function loadInput(data, name) {
  showError(null);
  let samples;
  try {
    samples = await decodeToMono(data);
  } catch {
    return showError("Couldn't read that file as audio. Try a wav, mp3, flac or ogg.");
  }
  const seconds = samples.length / SAMPLE_RATE;
  if (seconds < MIN_SECONDS || seconds > MAX_SECONDS) {
    return showError(`Samples must be between ${MIN_SECONDS} s and ${MAX_SECONDS / 60} min long; that one is ${seconds.toFixed(1)} s.`);
  }

  input = { name: name.replace(/\.[^.]+$/, "") || "sample", wav: encodeWav(samples) };
  const revealed = originalRow.hidden;
  animateHeight(glass, () => {
    original.load(samples, input.wav);
    originalRow.hidden = false;
    enhancedRow.hidden = true;
    enhanced.unload();
  });
  if (revealed) enter(originalRow);
}

uploadButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (file) await loadInput(await file.arrayBuffer(), file.name);
});

addEventListener("dragover", (event) => event.preventDefault());
addEventListener("drop", (event) => event.preventDefault());
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!busy) dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", (event) => {
  if (!dropzone.contains(event.relatedTarget)) dropzone.classList.remove("dragging");
});
dropzone.addEventListener("drop", async (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (file && !busy) await loadInput(await file.arrayBuffer(), file.name);
});

let recorder = null;

recordButton.addEventListener("click", async () => {
  showError(null);
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    return showError("Couldn't open a microphone. Check the browser's permission for this page.");
  }
  const chunks = [];
  const startedAt = performance.now();
  recorder = new MediaRecorder(stream);
  recorder.addEventListener("dataavailable", (event) => chunks.push(event.data));
  recorder.addEventListener("stop", async () => {
    clearInterval(timer);
    stream.getTracks().forEach((track) => track.stop());
    swap(recordingView, idleView);
    const blob = new Blob(chunks, { type: recorder.mimeType });
    await loadInput(await blob.arrayBuffer(), "recording");
  });
  const timer = setInterval(() => {
    const seconds = (performance.now() - startedAt) / 1000;
    recordingTime.textContent = formatTime(seconds);
    if (seconds >= MAX_SECONDS) recorder.stop();
  }, 250);
  recordingTime.textContent = "0:00";
  recorder.start();
  swap(idleView, recordingView);
});

stopButton.addEventListener("click", () => recorder?.state === "recording" && recorder.stop());

// ---------------------------------------------------------------------------------------------------------------------
// Enhancement

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

let progress = 0;

function setProgress(value, label) {
  progress = Math.max(progress, value);
  progressFill.style.width = `${(progress * 100).toFixed(1)}%`;
  enhanced.time.textContent = label ?? `${Math.round(progress * 100)}%`;
}

async function errorMessage(response) {
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((error) => error.msg).join("; ");
    if (detail?.message) return detail.message;
  } catch {}
  return `The API returned ${response.status} ${response.statusText}.`;
}

/**
 * Stream a job's latent previews into the background, and its sampling steps into the progress bar.
 *
 * @param {string} previewId The `previewId` returned with the job.
 * @returns {WebSocket} The open socket.
 */
function watchPreviews(previewId) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/preview/${previewId}`);
  socket.binaryType = "arraybuffer";
  socket.addEventListener("message", ({ data }) => {
    // Little-endian `struct.pack("<IIfHH", step, total, t, rows, columns)`, then row-major uint8 intensities
    const header = new DataView(data, 0, 16);
    const step = header.getUint32(0, true);
    const total = header.getUint32(4, true);
    const rows = header.getUint16(12, true);
    const columns = header.getUint16(14, true);
    background.showPreview(new Uint8Array(data, 16, rows * columns), rows, columns);
    setProgress(step / total);
  });
  return socket;
}

/**
 * Poll a job until it completes, updating the progress bar.
 *
 * @param {string} taskId The job's `taskId`.
 */
async function waitFor(taskId) {
  while (true) {
    const response = await fetch(`/status/${taskId}`);
    if (!response.ok) throw new Error(await errorMessage(response));
    const task = await response.json();
    if (task.status === "Completed") return;
    if (task.status === "Failed") throw new Error(task.error?.message || "Enhancement failed.");
    if (task.status === "Cancelled") throw new Error("Enhancement was cancelled.");
    setProgress(task.progress, task.status === "Created" ? "Queued" : undefined);
    await sleep(500);
  }
}

enhanceButton.addEventListener("click", async () => {
  if (!input || busy) return;
  setBusy(true);
  showError(null);
  progress = 0;

  const revealed = enhancedRow.hidden;
  animateHeight(glass, () => {
    enhanced.unload();
    enhanced.canvas.hidden = true;
    downloadLink.hidden = true;
    progressBar.hidden = false;
    enhancedRow.hidden = false;
    setProgress(0, "Queued");
  });
  if (revealed) enter(enhancedRow);

  let socket = null;
  try {
    const form = new FormData();
    form.append("file", input.wav, `${input.name}.wav`);
    form.append("out_format", "wav");
    form.append("guidance_scale", "5");
    form.append("guidance_target", "ssl");
    form.append("temperature", "0.9");
    form.append("num_sampling_steps", "32");
    const response = await fetch("/restore", { method: "POST", body: form });
    if (!response.ok) throw new Error(await errorMessage(response));
    const task = await response.json();
    if (task.previewId) socket = watchPreviews(task.previewId);

    await waitFor(task.taskId);
    setProgress(1);
    const download = await fetch(`/download/${task.taskId}`);
    if (!download.ok) throw new Error(await errorMessage(download));
    const blob = await download.blob();
    const samples = await decodeToMono(await blob.arrayBuffer());

    enhanced.load(samples, blob);
    downloadLink.href = enhanced.url;
    downloadLink.download = `${input.name}_enhanced.wav`;
    if (!reducedMotion.matches) await progressBar.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 200 }).finished;
    progressBar.hidden = true;
    enhanced.canvas.hidden = false;
    downloadLink.hidden = false;
    enhanced.draw();
    enter(enhanced.canvas, 0);
    enter(downloadLink, 60);
  } catch (error) {
    animateHeight(glass, () => (enhancedRow.hidden = true));
    showError(error.message);
  } finally {
    socket?.close();
    background.hidePreview();
    setBusy(false);
  }
});
