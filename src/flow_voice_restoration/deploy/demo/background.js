"use strict";

/**
 * Ken Perlin's improved noise, in 3D.
 */
class Perlin {
  constructor(seed = Math.random()) {
    const permutation = Array.from({ length: 256 }, (_, i) => i);
    let state = Math.floor(seed * 2 ** 31) || 1;
    for (let i = 255; i > 0; i--) {
      state = (state * 48271) % 2147483647;
      const j = state % (i + 1);
      [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
    }
    this.p = new Uint8Array(512);
    for (let i = 0; i < 512; i++) this.p[i] = permutation[i & 255];
  }

  static fade(t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  static lerp(t, a, b) {
    return a + t * (b - a);
  }

  static grad(hash, x, y, z) {
    const h = hash & 15;
    const u = h < 8 ? x : y;
    const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
  }

  /**
   * Noise at a point, roughly in [-1, 1].
   */
  noise(x, y, z) {
    const { fade, lerp, grad } = Perlin;
    const p = this.p;
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;
    const Z = Math.floor(z) & 255;
    x -= Math.floor(x);
    y -= Math.floor(y);
    z -= Math.floor(z);
    const u = fade(x);
    const v = fade(y);
    const w = fade(z);
    const A = p[X] + Y;
    const AA = p[A] + Z;
    const AB = p[A + 1] + Z;
    const B = p[X + 1] + Y;
    const BA = p[B] + Z;
    const BB = p[B + 1] + Z;
    return lerp(
      w,
      lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)), lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
      lerp(
        v,
        lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
        lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1)),
      ),
    );
  }
}

/**
 * A full-page grid of characters drawn from slowly drifting Perlin noise, like `TensorProgress`'s animation.
 * While a job is being enhanced, `showPreview()` crossfades the grid to the API's latent preview, in blue.
 */
class AsciiBackground {
  // Characters from quietest to loudest: `RAMPS["classic"]` in `utils/rich_logging.py`
  static RAMP = " .:-=+*#%@";
  // Barely-visible grey for the noise, and the accent blue for the preview, as [r, g, b, min alpha, max alpha]
  static NOISE_COLOR = [242, 242, 242, 0.05, 0.15];
  static PREVIEW_COLOR = [28, 160, 255, 0.35, 0.75];

  constructor(canvas, { fps = 12, transitionFps = 30, fontSize = 13, lineHeight = 16, speed = 0.00006 } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.fps = fps;
    this.transitionFps = transitionFps;
    this.fontSize = fontSize;
    this.lineHeight = lineHeight;
    this.speed = speed;
    this.perlin = new Perlin();
    this.reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
    this.mix = 0;
    this.mixTarget = 0;
    this.preview = null;
    this.drawnAt = -Infinity;
    this.tick = this.tick.bind(this);
  }

  async start() {
    await document.fonts.load(`${this.fontSize}px "IBM Plex Mono"`).catch(() => {});
    this.resize();
    addEventListener("resize", () => this.resize());
    requestAnimationFrame(this.tick);
  }

  resize() {
    const ratio = devicePixelRatio || 1;
    this.canvas.width = Math.round(innerWidth * ratio);
    this.canvas.height = Math.round(innerHeight * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.context.font = `${this.fontSize}px "IBM Plex Mono", ui-monospace, monospace`;
    this.context.textBaseline = "top";
    this.cellWidth = this.context.measureText("M").width;
    this.columns = Math.ceil(innerWidth / this.cellWidth);
    this.rows = Math.ceil(innerHeight / this.lineHeight);
    this.levels = new Uint8Array(this.rows * this.columns);
    this.previewTarget = new Float32Array(this.rows * this.columns);
    this.previewCurrent = new Float32Array(this.rows * this.columns);
    if (this.preview) {
      this.upsample();
      this.previewCurrent.set(this.previewTarget);
    }
    this.drawnAt = -Infinity;
  }

  /**
   * Show a latent preview frame, stretched over the whole grid.
   *
   * @param {Uint8Array} grid Row-major intensities, 0-255.
   * @param {number} rows Rows in `grid`.
   * @param {number} columns Columns in `grid`.
   */
  showPreview(grid, rows, columns) {
    const first = this.mixTarget === 0;
    this.preview = { grid: Uint8Array.from(grid), rows, columns };
    this.upsample();
    if (first) this.previewCurrent.set(this.previewTarget);
    this.mixTarget = 1;
  }

  /**
   * Fade back to the Perlin noise.
   */
  hidePreview() {
    this.mixTarget = 0;
  }

  /**
   * Bilinearly upsample the latest preview frame to the grid, into `previewTarget`.
   */
  upsample() {
    const { grid, rows, columns } = this.preview;
    for (let y = 0; y < this.rows; y++) {
      const sy = rows > 1 ? (y / Math.max(1, this.rows - 1)) * (rows - 1) : 0;
      const y0 = Math.floor(sy);
      const y1 = Math.min(rows - 1, y0 + 1);
      const fy = sy - y0;
      for (let x = 0; x < this.columns; x++) {
        const sx = columns > 1 ? (x / Math.max(1, this.columns - 1)) * (columns - 1) : 0;
        const x0 = Math.floor(sx);
        const x1 = Math.min(columns - 1, x0 + 1);
        const fx = sx - x0;
        const top = grid[y0 * columns + x0] * (1 - fx) + grid[y0 * columns + x1] * fx;
        const bottom = grid[y1 * columns + x0] * (1 - fx) + grid[y1 * columns + x1] * fx;
        this.previewTarget[y * this.columns + x] = (top * (1 - fy) + bottom * fy) / 255;
      }
    }
  }

  tick(now) {
    requestAnimationFrame(this.tick);
    const transitioning = this.mix !== this.mixTarget;
    if (now - this.drawnAt < 1000 / (transitioning ? this.transitionFps : this.fps)) return;
    const elapsed = Math.min(now - this.drawnAt, 250);
    this.drawnAt = now;

    // Ease the crossfade and the preview frames toward their targets, scaled to the time elapsed so they keep the
    // same speed (a third and 45% of the way per 12 fps frame) whatever the frame rate
    const frames = elapsed / (1000 / 12);
    this.mix += (this.mixTarget - this.mix) * (1 - (2 / 3) ** frames);
    if (Math.abs(this.mix - this.mixTarget) < 0.005) this.mix = this.mixTarget;
    if (this.preview) {
      const step = 1 - 0.55 ** frames;
      for (let i = 0; i < this.previewCurrent.length; i++) {
        this.previewCurrent[i] += (this.previewTarget[i] - this.previewCurrent[i]) * step;
      }
    }
    if (this.mix === 0) this.preview = null;

    this.draw(this.reducedMotion.matches ? 0 : now * this.speed);
  }

  draw(time) {
    const { context, columns, rows, cellWidth, lineHeight, mix, levels } = this;
    const ramp = AsciiBackground.RAMP;
    const top = ramp.length - 1;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        const i = y * columns + x;
        const n = this.perlin.noise(x * 0.045, y * 0.1, time) + 0.5 * this.perlin.noise(x * 0.11, y * 0.24, time * 1.7 + 31.4);
        let value = Math.min(1, Math.max(0, 0.5 + 0.75 * n));
        if (mix > 0) value = value * (1 - mix) + this.previewCurrent[i] * mix;
        levels[i] = Math.round(value * top);
      }
    }

    context.clearRect(0, 0, innerWidth, innerHeight);
    const [nr, ng, nb, nLow, nHigh] = AsciiBackground.NOISE_COLOR;
    const [pr, pg, pb, pLow, pHigh] = AsciiBackground.PREVIEW_COLOR;
    for (let level = 1; level <= top; level++) {
      const intensity = level / top;
      const r = Math.round(nr + (pr - nr) * mix);
      const g = Math.round(ng + (pg - ng) * mix);
      const b = Math.round(nb + (pb - nb) * mix);
      const alpha = (nLow + (nHigh - nLow) * intensity) * (1 - mix) + (pLow + (pHigh - pLow) * intensity) * mix;
      context.fillStyle = `rgb(${r} ${g} ${b} / ${alpha.toFixed(3)})`;
      const character = ramp[level];
      for (let i = 0; i < levels.length; i++) {
        if (levels[i] === level) {
          context.fillText(character, (i % columns) * cellWidth, Math.floor(i / columns) * lineHeight);
        }
      }
    }
  }
}
