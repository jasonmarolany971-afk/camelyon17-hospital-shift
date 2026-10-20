"""Gradio demo: metastasis detection on lymph-node patches, with a confidence check and a heat map.

Run locally (from the repository root):  python app/app.py
On Hugging Face Spaces this folder is the whole Space (it does not import the project's src/).

Model: frozen Phikon ViT-B/16 (Owkin, non-commercial license) + a linear head trained on
Camelyon17-WILDS (3 training hospitals), chosen on the validation hospital. Everything the app
needs comes from model/head.json and model/card.json, written by `python -m src.export_app`.

Research tool, not a medical device.
"""

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
HEAD = json.loads((HERE / "model" / "head.json").read_text())
CARD = json.loads((HERE / "model" / "card.json").read_text())
PATCH = HEAD["preprocessing"]["input_size"]        # 96: size of the training patches
MODEL_SIZE = HEAD["preprocessing"]["model_size"]   # 224: size Phikon expects
DISPLAY = 384                                      # size of the images shown
CAM_COLOR = np.array([0xED, 0xA1, 0x00]) / 255     # amber, as in the project's Grad-CAM figure
REPO_URL = "https://github.com/jasonmarolany971-afk/camelyon17-hospital-shift"


def to_patch(image: Image.Image) -> tuple[Image.Image, str]:
    """RGB patch of the training size; any other image is center-cropped to a square and resized."""
    image = image.convert("RGB")
    if image.size == (PATCH, PATCH):
        return image, ""
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    patch = image.crop((left, top, left + side, top + side)).resize((PATCH, PATCH), Image.BILINEAR)
    note = (f"Your image ({width} x {height}) was center-cropped and resized to {PATCH} x {PATCH}. The model was "
            "trained on 96 x 96 Camelyon17-WILDS patches: at another magnification its output is unreliable.")
    return patch, note


def preprocess(patch: Image.Image) -> torch.Tensor:
    """Same as the feature extraction: bilinear resize to 224, then ImageNet normalization."""
    array = np.asarray(patch.resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    mean = np.array(HEAD["preprocessing"]["mean"], dtype=np.float32)
    std = np.array(HEAD["preprocessing"]["std"], dtype=np.float32)
    return torch.from_numpy(((array - mean) / std).transpose(2, 0, 1)).unsqueeze(0)


class PhikonClassifier:
    """Frozen Phikon, standardized [CLS] token, linear head; Grad-CAM on the input of the last block.

    The final [CLS] token does not depend on the patch tokens of its own layer, so the gradient is
    taken at the input of the last block, where attention mixes the tokens: a 14 x 14 grid.
    The explained score is the logit difference (tumor - normal), i.e. the decision itself.
    """

    def __init__(self, vit: torch.nn.Module, head: dict):
        self.vit = vit.eval()
        self.mean = torch.tensor(head["feature_mean"])
        self.std = torch.tensor(head["feature_std"])
        self.weight = torch.tensor(head["weight"])  # rows: normal, tumor
        self.bias = torch.tensor(head["bias"])
        layers = vit.layers if hasattr(vit, "layers") else vit.encoder.layer  # transformers 5 / 4
        layers[-1].layernorm_before.register_forward_hook(self._keep)
        self.tokens = None

    def _keep(self, module, inputs, output):
        self.tokens = output

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(logit difference tumor - normal, Grad-CAM maps of shape batch x 14 x 14)."""
        with torch.enable_grad():
            cls = self.vit(pixel_values=x).last_hidden_state[:, 0]
            logits = ((cls - self.mean) / self.std) @ self.weight.T + self.bias
            z = logits[:, 1] - logits[:, 0]
            (grads,) = torch.autograd.grad(z.sum(), self.tokens)
        tokens, grads = self.tokens[:, 1:].detach(), grads[:, 1:]  # drop the [CLS] token
        weights = grads.mean(dim=1, keepdim=True)                   # one weight per channel
        maps = torch.relu((tokens * weights).sum(dim=-1))
        side = int(round(maps.shape[1] ** 0.5))
        return z.detach(), maps.reshape(-1, side, side)


@lru_cache(maxsize=1)
def load_model() -> PhikonClassifier:
    from transformers import ViTModel

    backbone = HEAD["backbone"]
    vit = ViTModel.from_pretrained(backbone["hf_id"], revision=backbone["revision"], add_pooling_layer=False)
    return PhikonClassifier(vit, HEAD)


def overlay(patch: Image.Image, cam: np.ndarray) -> Image.Image:
    """Patch enlarged for display; amber where the map is high (each map scaled to its own maximum)."""
    rgb = np.asarray(patch.resize((DISPLAY, DISPLAY), Image.BILINEAR), dtype=np.float32) / 255.0
    if cam.max() > 0:
        alpha = np.asarray(Image.fromarray((cam / cam.max()).astype(np.float32)).resize((DISPLAY, DISPLAY),
                                                                                      Image.NEAREST))[..., None]
        rgb = rgb * (1 - 0.6 * alpha) + CAM_COLOR * 0.6 * alpha
    image = Image.fromarray((rgb * 255).round().astype(np.uint8))
    third = DISPLAY // 3  # the label describes the central 32 x 32 region of the 96 x 96 patch
    ImageDraw.Draw(image).rectangle((third, third, 2 * third, 2 * third), outline=(255, 255, 255), width=2)
    return image


def analyze(image):
    if image is None:
        return "Upload a patch or click an example.", None, None
    patch, note = to_patch(image)
    z, cams = load_model()(preprocess(patch))
    z, cam = float(z[0]), cams[0].numpy()
    probability = 1 / (1 + np.exp(-z / HEAD["temperature"]))   # calibrated on the validation hospital
    if abs(z) < HEAD["referral_abs_logit"]:
        verdict = ("## Review by a pathologist\nThe model is not confident enough on this patch: it refers the "
                   "least confident patches (20% of those of the validation hospital).")
    elif z > 0:
        verdict = "## Tumor detected"
    else:
        verdict = "## No tumor detected"
    verdict += (f"\n\nTumor score: **{probability:.0%}**. This number is calibrated on one hospital only; on "
                "another hospital it is a ranking score, not an exact probability.")
    if cam.max() == 0:
        verdict += "\n\nHeat map empty: no region of the patch pushes towards 'tumor'."
    if note:
        verdict += f"\n\n*{note}*"
    return verdict, {"tumor": probability, "normal": 1 - probability}, overlay(patch, cam)


def about() -> str:
    run, referral = CARD["this_run"], CARD["referral"]
    low, high = CARD["method_test_accuracy_ci95_over_patients"]
    return f"""
**What it does.** Classifies a 96 x 96 patch of a lymph-node slide (H&E) as containing breast-cancer
metastasis or not, and refers uncertain patches to a pathologist. The heat map shows which parts of the
patch push the decision towards "tumor" (Grad-CAM on the last transformer block, 14 x 14 grid; the white
square is the central region the training labels describe).

**Model.** Frozen [Phikon]({"https://huggingface.co/" + HEAD["backbone"]["hf_id"]}) (ViT-B/16 pretrained on
histology) + a linear layer trained on 3 hospitals of Camelyon17-WILDS. Chosen on the validation hospital only;
the test hospital was never used to choose anything. Seed {CARD["selection"]["chosen_seed"]} of 4.

**Results of this model** (patch accuracy): training hospitals {run["id_val"]["accuracy"]:.1%}; validation
hospital (unseen in training) {run["val"]["accuracy"]:.1%}; test hospital (unseen) {run["test"]["accuracy"]:.1%},
AUC {run["test"]["auc"]:.3f}. Over 4 seeds, 95% interval on the test hospital when its patients are redrawn:
{low:.1%} to {high:.1%}. With referral, on the test hospital the model answers {referral["test"]["answered"]:.0%}
of patches and its error drops from {referral["test"]["error_without_referral"]:.1%} to
{referral["test"]["error_when_answered"]:.1%}.

**Limits.** Two unseen hospitals only, 8 to 9 patients each. The displayed score is calibrated on one
hospital (expected calibration error on the test hospital {CARD["temperature"]["ece_test_before_after"][1]:.3f}).
Patches only, not whole slides. Labels come from hand-drawn contours and are themselves imperfect.

**Licenses.** Data: Camelyon17 (CC0). Phikon weights: Owkin non-commercial license, so this demo is for
non-commercial research and education only. Code and full analysis: [GitHub]({REPO_URL}).
"""


def build_demo() -> gr.Blocks:
    examples = CARD["examples"]
    with gr.Blocks(title="Metastasis detection across hospitals") as demo:
        gr.Markdown("# Metastasis detection across hospitals\n"
                    "**Research tool, not a medical device. Do not use for diagnosis.**")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="H&E patch (96 x 96 pixels)", sources=["upload", "clipboard"])
                button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                scores = gr.Label(label="Scores", num_top_classes=2)
                heat_map = gr.Image(label="Heat map (amber: pushes towards tumor)", type="pil")
        gr.Examples(examples=[[str(HERE / "examples" / e["file"])] for e in examples], inputs=image,
                    outputs=[verdict, scores, heat_map], fn=analyze, run_on_click=True, cache_examples=False,
                    example_labels=[f"Hospital {e['hospital']} ({e['role']}), {e['label']}" for e in examples],
                    label="Examples: one tumor and one normal patch per hospital, drawn at random")
        with gr.Accordion("About this model", open=False):
            gr.Markdown(about())
        button.click(analyze, inputs=image, outputs=[verdict, scores, heat_map])
    return demo


if __name__ == "__main__":
    build_demo().launch()
