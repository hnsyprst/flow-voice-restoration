"use strict";

// TEMPORARY: sliders for tuning the panel's liquid glass. Delete this file and its <script> tag once the values are
// copied into `liquidOptions` (panel.js) and style.css.

(() => {
  const css = `
    .tuner { position: fixed; right: 16px; bottom: 16px; z-index: 10; width: 280px; padding: 12px 14px;
      border: 1px solid rgb(255 255 255 / 15%); border-radius: 10px; background: rgb(12 13 15 / 88%);
      font: 400 11px/1.4 var(--mono); color: var(--silver); }
    .tuner header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; color: var(--bright); }
    .tuner label { display: grid; grid-template-columns: 76px 1fr 44px; align-items: center; gap: 8px; margin: 4px 0; }
    .tuner input[type=range] { width: 100%; accent-color: var(--bright); }
    .tuner output { text-align: right; color: var(--bright); font-variant-numeric: tabular-nums; }
    .tuner .note { margin: 6px 0; color: var(--error); }
    .tuner .buttons { display: flex; gap: 6px; margin-top: 10px; }
    .tuner button { flex: 1; height: 26px; border: 1px solid rgb(255 255 255 / 20%); border-radius: 999px;
      background: transparent; color: var(--bright); font: inherit; cursor: pointer; }
    .tuner button:hover { background: rgb(255 255 255 / 6%); }
    .tuner pre { margin: 8px 0 0; white-space: pre-wrap; color: var(--bright); user-select: all; }
    .tuner.collapsed label, .tuner.collapsed .buttons, .tuner.collapsed pre, .tuner.collapsed .note { display: none; }
    .tuner.collapsed header { margin: 0; }
  `;

  // [key, label, min, max, step, initial]: `glass` keys are `liquidGlass()` options, `tint` keys are style.css values
  const controls = [
    ["scale", "scale", -200, 0, 1, liquidOptions.scale ?? -112],
    ["chroma", "chroma", 0, 20, 0.5, liquidOptions.chroma ?? 6],
    ["border", "border", 0, 0.3, 0.005, liquidOptions.border ?? 0.07],
    ["mapBlur", "map blur", 0, 40, 1, liquidOptions.mapBlur ?? 12],
    ["blur", "blur", 0, 20, 0.5, liquidOptions.blur ?? 3],
    ["saturate", "saturate", 0, 3, 0.05, liquidOptions.saturate ?? 1.5],
    ["bodyTint", "body tint %", 0, 100, 1, 30],
    ["rowTint", "row tint %", 0, 100, 1, 15],
    ["highlight", "highlight %", 0, 100, 1, 50],
  ];
  const values = Object.fromEntries(controls.map(([key, , , , , initial]) => [key, initial]));

  const style = document.createElement("style");
  style.textContent = css;
  document.head.append(style);

  const tuner = document.createElement("aside");
  tuner.className = "tuner";
  tuner.innerHTML = `<header><span>Glass tuner</span><button type="button" data-action="collapse" style="flex:none;width:26px">–</button></header>`;
  if (!liquid.supported) {
    tuner.insertAdjacentHTML("beforeend", `<p class="note">No refraction in this browser (Chromium only): only blur, saturate and the tints apply.</p>`);
  }
  for (const [key, label, min, max, step] of controls) {
    tuner.insertAdjacentHTML(
      "beforeend",
      `<label>${label}<input type="range" name="${key}" min="${min}" max="${max}" step="${step}" value="${values[key]}"><output>${values[key]}</output></label>`,
    );
  }
  tuner.insertAdjacentHTML(
    "beforeend",
    `<div class="buttons"><button type="button" data-action="copy">Copy values</button><button type="button" data-action="reset">Reset</button></div><pre></pre>`,
  );
  document.body.append(tuner);
  const summary = tuner.querySelector("pre");

  function describe() {
    const { scale, chroma, border, mapBlur, blur, saturate, bodyTint, rowTint, highlight } = values;
    return (
      `liquidOptions = { scale: ${scale}, chroma: ${chroma}, border: ${border}, mapBlur: ${mapBlur}, blur: ${blur}, saturate: ${saturate} }\n` +
      `.panel-body tint ${bodyTint}%, .row tint ${rowTint}%, highlight ${highlight}%`
    );
  }

  let pending = false;
  function apply() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const { bodyTint, rowTint, highlight, ...options } = values;
      Object.assign(liquidOptions, options, { fallbackBlur: options.blur });
      liquid.destroy();
      liquid = liquidGlass(glass, liquidOptions);
      document.getElementById("dropzone").style.backgroundColor = `rgb(29 30 33 / ${bodyTint}%)`;
      document.querySelectorAll(".row").forEach((row) => (row.style.background = `rgb(0 0 0 / ${rowTint}%)`));
      glass.style.boxShadow = `inset 0 1px 1px rgb(255 255 255 / ${highlight}%), inset 0 -8px 20px rgb(255 255 255 / 6%)`;
      summary.textContent = describe();
    });
  }

  tuner.addEventListener("input", (event) => {
    const input = event.target;
    values[input.name] = Number(input.value);
    input.nextElementSibling.value = input.value;
    apply();
  });

  tuner.addEventListener("click", async (event) => {
    const action = event.target.dataset?.action;
    if (action === "collapse") {
      tuner.classList.toggle("collapsed");
      event.target.textContent = tuner.classList.contains("collapsed") ? "+" : "–";
    } else if (action === "copy") {
      try {
        await navigator.clipboard.writeText(describe());
        event.target.textContent = "Copied";
      } catch {
        event.target.textContent = "Select the text below";
      }
      setTimeout(() => (event.target.textContent = "Copy values"), 1500);
    } else if (action === "reset") {
      for (const [key, , , , , initial] of controls) {
        values[key] = initial;
        const input = tuner.querySelector(`input[name=${key}]`);
        input.value = initial;
        input.nextElementSibling.value = initial;
      }
      apply();
    }
  });

  summary.textContent = describe();
})();
