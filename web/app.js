// In-browser demo: the ONNX model returns the decision z = logit(tumor) - logit(normal) and its
// 3 x 3 Grad-CAM map. Everything shown comes from settings.json (written by src/export_web_demo.py).

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const AMBER = [237, 161, 0];
let settings, session;

const pct = (x, digits = 1) => `${(100 * x).toFixed(digits)}%`;

function toTensor(canvas) {
  const size = settings.input.size;
  const { data } = canvas.getContext("2d").getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    for (let c = 0; c < 3; c++) {
      tensor[c * size * size + i] = (data[4 * i + c] / 255 - settings.input.mean[c]) / settings.input.std[c];
    }
  }
  return new ort.Tensor("float32", tensor, [1, 3, size, size]);
}

function toPatch(image) {
  // Training size: kept as is. Other images: center-cropped to a square and resized, with a warning.
  const size = settings.input.size;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const w = image.naturalWidth, h = image.naturalHeight;
  let note = "";
  if (w === size && h === size) {
    context.drawImage(image, 0, 0);
  } else {
    const side = Math.min(w, h);
    context.imageSmoothingQuality = "high";
    context.drawImage(image, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);
    note = `Your image (${w} x ${h}) was center-cropped and resized to ${size} x ${size}. The model was trained on ` +
           `${size} x ${size} Camelyon17-WILDS patches: at another magnification its output is unreliable.`;
  }
  return { canvas, note };
}

async function predict(patch) {
  const output = await session.run({ image: toTensor(patch) });
  return { z: output.z.data[0], cam: Array.from(output.cam.data) };
}

function draw(patch, cam) {
  const view = document.getElementById("view");
  const context = view.getContext("2d");
  const side = view.width, cell = side / 3, max = Math.max(...cam);
  context.imageSmoothingEnabled = true;
  context.drawImage(patch, 0, 0, side, side);
  if (max > 0) {
    cam.forEach((value, k) => {
      context.fillStyle = `rgba(${AMBER.join(",")}, ${(0.6 * value / max).toFixed(3)})`;
      context.fillRect((k % 3) * cell, Math.floor(k / 3) * cell, cell, cell);
    });
  }
  context.strokeStyle = "#ffffff";
  context.lineWidth = 2;
  context.strokeRect(cell, cell, cell, cell);
}

function show({ z, cam }, note) {
  const verdict = document.getElementById("verdict");
  const probability = 1 / (1 + Math.exp(-z / settings.temperature));  // calibrated on the validation hospital
  const referred = Math.abs(z) < settings.referral_abs_logit;
  verdict.className = referred ? "refer" : "";
  verdict.textContent = referred ? "Review by a pathologist" : (z > 0 ? "Tumor detected" : "No tumor detected");
  document.getElementById("bar").style.width = pct(probability);
  document.getElementById("score").textContent =
    `Tumor score: ${pct(probability, 0)}. Calibrated on one hospital only; on another hospital it is a ranking ` +
    `score, not an exact probability.` +
    (referred ? ` The model refers its least confident patches (${pct(1 - settings.referral.val.answered, 0)} of those ` +
                `of the validation hospital).` : "") +
    (Math.max(...cam) === 0 ? " Heat map empty: no region pushes towards tumor." : "");
  document.getElementById("note").textContent = note;
}

async function analyzeImage(image) {
  const { canvas, note } = toPatch(image);
  const result = await predict(canvas);
  draw(canvas, result.cam);
  show(result, note);
  return result;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

// Used by the project's browser check (src/check_web_demo.py results are compared with these values).
window.demoAnalyze = async (url) => analyzeImage(await loadImage(url));

function exampleLabel(e) {
  if (e.kind === "referred") return `Referred by the model (hospital ${e.hospital}, validation), truly ${e.label}`;
  if (e.kind === "error") return `An error (hospital ${e.hospital}, test), truly ${e.label}`;
  return `Hospital ${e.hospital} (${e.role}), ${e.label}`;
}

function renderExamples() {
  const container = document.getElementById("examples");
  for (const e of settings.examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<img src="examples/${e.file}" alt=""><span></span>`;
    button.querySelector("span").textContent = exampleLabel(e);
    button.addEventListener("click", () => window.demoAnalyze(`examples/${e.file}`));
    container.appendChild(button);
  }
}

function renderAbout() {
  const run = settings.this_run, test = settings.referral.test;
  const [low, high] = settings.method_test_accuracy_ci95_over_patients;
  document.getElementById("about").innerHTML = `
    <p><strong>What it does.</strong> Classifies a 96 x 96 patch of a lymph-node slide (H&amp;E) as containing
    breast-cancer metastasis or not, refers uncertain patches to a pathologist, and shows which parts of the patch
    push the decision towards tumor. Everything runs in your browser: no image leaves your computer.</p>
    <p><strong>Model.</strong> ${settings.model.description}, trained on 3 hospitals of Camelyon17-WILDS. It is
    the best CNN on the validation hospital, and seed ${settings.model.chosen_seed} of 4 was chosen the same way;
    the test hospital was never used to choose anything. It is small enough to run in a browser; the project's best
    method on the validation hospital (frozen Phikon + linear layer) is too large for that and runs locally.</p>
    <p><strong>Results of this model</strong> (patch accuracy): training hospitals ${pct(run.id_val.accuracy)};
    validation hospital, unseen in training, ${pct(run.val.accuracy)}; test hospital, unseen,
    ${pct(run.test.accuracy)} (AUC ${run.test.auc.toFixed(3)}). Over 4 seeds, the method reaches
    ${pct(settings.method_test_accuracy_mean_over_seeds)} on the test hospital, 95% interval ${pct(low)} to
    ${pct(high)} when its patients are redrawn. With referral, on the test hospital the model answers
    ${pct(test.answered, 0)} of patches and its error goes from ${pct(test.error_without_referral)} to
    ${pct(test.error_when_answered)}.</p>
    <p><strong>Limits.</strong> Two unseen hospitals only. The score is calibrated on one hospital (expected
    calibration error on the test hospital ${settings.ece_test_before_after[1].toFixed(3)}). Patches only, not
    whole slides. Labels come from hand-drawn contours and are themselves imperfect.</p>
    <p><strong>Code, results and licenses:</strong>
    <a href="https://github.com/jasonmarolany971-afk/camelyon17-hospital-shift">GitHub</a>. Code and weights: MIT.
    Data: Camelyon17 (CC0).</p>`;
}

async function main() {
  settings = await (await fetch("settings.json")).json();
  renderExamples();
  renderAbout();
  session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
  document.getElementById("verdict").textContent = "Upload a patch or click an example.";
  document.getElementById("file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) await analyzeImage(await loadImage(URL.createObjectURL(file)));
  });
  window.demoReady = true;
}

main().catch((error) => {
  document.getElementById("verdict").textContent = `Could not load the model: ${error.message}`;
});
