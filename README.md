<div align="center">

# EnergyCons

Custom (Transparent) Lifting Wavelet Transform (LWT) for Energy & Multivariate Time‑Series Compression + Neural Network Acceleration

</div>

## 1. Why This Project Exists
Traditional wavelet workflows in Python normally rely on PyWavelets (excellent, but a black box if you want to tinker with lifting steps). In research / constrained‑edge scenarios we often need:

* Full transparency into every split / predict / update step
* The ability to modify lifting coefficients or boundary handling on the fly
* Embedding wavelet decomposition directly as a learnable or structural “feature extractor layer” in deep networks to shrink intermediate representations and FLOPs
* Extensibility from 1‑D signals → arbitrary N‑D tensors (e.g., (H,W), (C,H,W), (T,H,W), …)

EnergyCons provides a from‑scratch, pure‑NumPy / pure‑PyTorch / pure‑TensorFlow set of Lifting Wavelet Transform utilities (currently Haar + a simple biorthogonal variant) that you can:
1. Use to reduce dataset size (retain approximation coefficients, optionally discard / quantize details)
2. Plug into neural network pipelines (see `lwt_model.py`) as a replacement for early convolutional layers, reducing parameters and compute
3. Scale to N dimensions without PyWavelets, retaining modifiability & educational clarity

> No dependency on PyWavelets is deliberate: every operation is owned, inspectable, and easy to fork.

---

## 2. Repository Layout (Key Files)

| File / Dir | Purpose |
|------------|---------|
| `cwnn_model.py` | Baseline 1‑D Conv + dense regression model (comparison target). |
| `lwt_model.py` | PyTorch model that stacks custom 1‑D LWT levels then feeds FC layers. |
| `dataset.py` | Windowed in‑memory dataset loader for REEDD‑style appliance CSVs (with normalization & active threshold logic). |
| `train_and_run.py` | Unified training script: selects CWNN or LWT model per `settings.yaml`. |
| `nd_lwt.py` | Experimental N‑D LWT + TensorFlow dense auto‑regressor (interactive; asks for shape & wavelet type). |
| `UI/` | Flask app + N‑D Haar lifting utilities for uploading CSVs & exporting coefficients. |
| `UI/wavelet_transform.py` | General Haar LWT N‑D (decompose / reconstruct) – building blocks for web UI. |
| `settings.yaml` | Global & per‑appliance hyperparameters (window length, levels, learning rate, etc.). |
| `template.yaml` | Example appliance / building structure template (metadata scaffold). |
| `generated_coeffs/` | Stored coefficient CSVs produced offline or via UI. |
| `uploads/` | Sample / user‑uploaded CSV inputs for UI. |
| `requirements.txt` | Python dependencies (minimal, no PyWavelets). |

---

## 3. Core Idea: Lifting Wavelet Transform (Haar Variant)

For a 1‑D signal x:
1. Split: even samples e, odd samples o
2. Predict: detail d = o − e (Haar prediction)
3. Update: approximation a = e + d/2 (preserves mean)

Inverse does: e = a − d/2; o = d + e; then interleave.

Stacking levels recursively halves (approximately) the spatial/temporal length each time. In N‑D we apply the 1‑D lifting along each axis in sequence, labeling sub‑bands with L/H patterns (e.g., LL, LH, HL, HH for 2‑D; generalizes to 2^N bands per level). Approximation (all ‘L’) feeds the next level.

### Why This Helps Models
* Early compression shrinks sequence length before dense / conv layers → fewer parameters.
* Detail coefficients can be optionally pruned or quantized for speed / memory trade‑offs.
* Deterministic, invertible transform keeps reconstruction error analytically bounded (MSE measured in scripts).

### Why Not Use Convs First?
Convolutions learn filters; lifting gives mathematically structured, multi‑resolution features without learning extra weights, reducing overfitting risk in small‑data regimes.

---

## 4. LWT in a PyTorch Model (`lwt_model.py`)

`LWTBasedModel`:
* Applies `num_lwt_levels` of 1‑D LWT (Haar‑like) to each input window
* Concatenates all detail coefficients + final approximation
* Feeds to fully connected layers for regression

This replaces conv/pool stacks in `CWNN` while achieving similar representational hierarchy with fewer learned operations.

---

## 5. N‑Dimensional Extension

Two independent implementations illustrate N‑D scaling:

1. `UI/wavelet_transform.py`: Focused, production‑style Haar N‑D (deterministic; shape bookkeeping for exact inversion). Used by the Flask UI.
2. `nd_lwt.py`: Research playground implementing both Haar and a simplified `linear_bior` variant, plus a TensorFlow dense predictor on compressed coefficients.

Both follow: iterative axis‑wise lifting per level, track original shapes, reconstruct by reversing axis order.

---

## 6. Installation

### Quick
```bash
pip install -r requirements.txt
```

If you only need PyTorch path and not the TensorFlow experiment (`nd_lwt.py`), you can remove `tensorflow` from `requirements.txt` for a lighter install.

Optional (CPU‑only TF): replace `tensorflow` with `tensorflow-cpu`.

---

## 7. Datasets
Expected folder layout for appliance energy data (REDD‑style):
```
dataset/
	redd_house1_0.csv
	redd_house1_1.csv
	...
```
Each CSV must contain at least columns: `main` and the appliance column (e.g., `dishwasher`, `fridge`, `microwave`). Filenames are filtered by building prefixes listed in `settings.yaml` (e.g., `redd_house2`).

`InMemoryKoreaDataset` performs:
* Window slicing (size L)
* Optional normalization (global mean/std of main & target)
* Active / inactive labeling via `active_threshold`

---

## 8. Configuration (`settings.yaml`)

Global hyperparameters:
```yaml
hparams:
	lr: 0.001
	batch_size: 64
	epochs: 5
```
Per‑appliance block selects model (`CWNN` or `LWT`), window length `L`, hidden size `H`, and (if LWT) `num_lwt_levels`.

Example (excerpt):
```yaml
microwave:
	model: LWT
	hparams:
		L: 128
		H: 1024
		num_lwt_levels: 4
```

Switching models is as simple as changing `model:`.

---

## 9. Training a Model

Run (arguments: settings file, dataset folder, appliance key):
```bash
python train_and_run.py settings.yaml dataset microwave
```
During training you’ll see loss per epoch and ETA. Device auto‑selects CUDA if available.

Outputs (currently) are printed; you can extend to checkpoint saving via the utilities in `utils.py` (`save_model`).

---

## 10. Using the Flask UI for Coefficients

Launch the app (from repo root):
```bash
python UI/app.py
```
Then open `http://127.0.0.1:5000/`:
1. Upload a CSV
2. Choose decomposition levels
3. Select 1‑D (single numeric column) or 2‑D (all numeric columns) mode
4. Download generated coefficient CSV from `generated_coeffs/`

The UI verifies reconstruction fidelity (`np.allclose`).

---

## 11. Research Script: `nd_lwt.py`

Interactive terminal prompts:
1. Enter N‑D target shape (e.g., `32,32` or `4,16,16`)
2. Choose wavelet (`haar` or `linear_bior` simplified)
3. Performs forward multi‑level LWT on Time / Speed / Reduced columns
4. Trains a dense TensorFlow model to map compressed X→Y coefficients
5. Reconstructs and reports MSE for original vs. reconstructed and predicted signals

Use this to experiment with dimensional reshaping & compression ratios.

---

## 12. Performance & Compression Considerations

| Aspect | Benefit | Notes |
|--------|---------|-------|
| Parameter Count | No learnable weights in lifting layers | Stable inductive bias |
| Latency | Reduced input length for dense layers | Especially large `L` windows |
| Memory | Optionally discard/prune detail bands | Keep indices if selective retention |
| Interpretability | Explicit detail vs. approximation separation | Facilitates frequency‑aware pruning |

You can measure compression ratio after `k` levels roughly as: 1 / 2^k for 1‑D approximation stream (ignoring details). For N‑D, approximation volume shrinks by 1 / 2^(k*N) (before considering details kept/discarded).

---

## 13. Extending / Customizing

Ideas:
* Replace fixed 0.5 update with learnable scalar(s) (constrained for invertibility)
* Integrate attention over coefficient bands
* Mixed precision storage for detail coefficients
* Adaptive level depth per sample (early exit if energy below threshold)
* Plug into sequence models as a pre‑tokenizer for transformers

---

## 14. Limitations
* Currently only Haar + a basic linear biorthogonal variant (research script) – no full wavelet family catalogue
* No integrated evaluation metrics beyond MSE / MAE
* Minimal error handling in the interactive script (`nd_lwt.py`) for mis‑sized shapes
* No batching pipeline for N‑D TensorFlow example (single sample demo)

---

## 15. Roadmap (Potential Next Steps)
* Add PyTorch autograd unit tests verifying exact inverse
* Add coefficient energy thresholding + sparsity stats
* Provide benchmarking harness (wall time vs. CNN baseline)
* Add ONNX export path using pure ops
* Optional PyWavelets interop layer for validation comparisons

---

## 16. Contributing
PRs welcome for:
* Additional wavelet families (lifting coefficients table)
* Vectorized N‑D inverse simplification
* Documentation clarifications & examples

Please open an issue first for architectural changes.

---

## 17. Citation (If Helpful in Academic Context)
```
@software{energycons_lwt,
	title  = {EnergyCons: Transparent N-D Lifting Wavelet Transform for Energy Time-Series Modeling},
	author = {Contributors},
	year   = {2025},
	url    = {https://github.com/Champion2049/EnergyCons}
}
```

---

## 18. License
MIT License is present with the authors being Chirayu Nilesh Chaudhari & Rtamanyu NJ.

---

## 19. Quick Start Recap
```bash
# Install
pip install -r requirements.txt

# Train (PyTorch, LWT model)
python train_and_run.py settings.yaml dataset microwave

# Run coefficient UI
python UI/app.py
```

---

Questions / ideas? Open an issue or start a discussion. Enjoy exploring transparent wavelet lifting!
