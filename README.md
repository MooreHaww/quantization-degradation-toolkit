# quantization-toolkit (`qtk`)

From-scratch weight quantizers in PyTorch, plus three experiments on how weight
precision degrades EleutherAI Pythia models (70m / 160m / 410m).

**What this repo tests.** My paper on quantization-induced degradation and
precision-limited in-context learning is a theory paper with no experiments. One
of its predictions is about weight-only quantization: as weight bit-width drops,
model quality should degrade *smoothly*, with no cliff at 3 bits or more. This
repo tests that prediction empirically. It does **not** reproduce results from
the paper.

The three experiments:

1. **Reconstruction error** (`qtk error`): how far the fake-quantized weight
   matrices are from the originals, as a function of bit-width and quantization
   granularity.
2. **Perplexity** (`qtk ppl`): WikiText-2 perplexity at FP32, FP16 and 8/6/4/3/2-bit weights.
3. **In-context learning** (`qtk icl`): SST-2 few-shot accuracy at k = 0, 1, 4, 8, 16
   shots for each precision. *Does lower precision reduce how much the model gains
   from extra shots?*

Each experiment writes a CSV, a PNG and a `*_meta.json` (arguments, library
versions, device, git SHA) to `results/`.

## Setup (Windows 11, PowerShell)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# GPU: the RTX 50-series (Blackwell, sm_120) needs a PyTorch build for CUDA 12.8 or newer.
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU only: pip install torch

pip install -e ".[dev]"

# check the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If no cu128 wheel exists for your Python version, try the `cu129` or `cu130`
index, or create the venv with Python 3.12. On Linux/macOS, activate with
`source .venv/bin/activate`; everything else is the same.

The first run downloads the models and datasets to the Hugging Face cache
(roughly 1–2 GB for the three models).

## Running

```powershell
pytest                           # unit + offline end-to-end tests (no downloads)
qtk all --quick                  # tiny smoke test of all three experiments -> results/quick/
qtk error                        # Exp 1
qtk ppl                          # Exp 2
qtk icl --seeds "0,1,2,3,4"      # Exp 3 (final run: 5 demo seeds)
```

**PowerShell note:** quote comma-separated lists (`--bits "8,4,2"`). PowerShell
turns an unquoted `8,4,2` into an array and splits it into separate arguments.

Every command takes `--device auto|cpu|cuda` (default `auto`: CUDA if available),
`--out-dir` (default `results`), `--seed` (default 0) and `--revision`
(default `step143000`). Run `qtk <command> --help` for all flags.

Quick-run flags for CPU:

| command | flag | effect |
|---|---|---|
| `qtk error` | `--max-layers N` | only the first N `nn.Linear` layers per model |
| `qtk ppl` | `--max-chunks N` | only the first N 1024-token chunks |
| `qtk icl` | `--n-eval N`, `--shots`, `--seeds` | fewer examples / conditions |
| all | `--models`, `--bits` | e.g. `--models pythia-70m --bits "8,2"` |

Examples:

```powershell
qtk ppl --models pythia-70m --bits "8,4,2" --max-chunks 8 --device cpu
qtk icl --models pythia-70m --bits "8,4" --shots "0,4" --seeds 0 --n-eval 64
qtk ppl --scheme sym --granularity channel --out-dir results/sym_channel
```

## Quantizers

Implemented from scratch in [`src/qtk/quant.py`](src/qtk/quant.py), with no
quantization libraries. Any bit-width from 2 to 8 is supported. All quantizer
arithmetic runs in **float32**, whatever the model's dtype.

A weight `W` of shape `[out, in]` is split into groups, each with its own scale:

| granularity | one scale per |
|---|---|
| `tensor` | whole matrix |
| `channel` | output row |
| `group:N` (N = 32, 64, 128, ...) | N contiguous input columns within a row. If `in` is not a multiple of N, the last group is zero-padded; the padding never changes the range. |

| scheme | definition (b bits, round = round-half-to-even) |
|---|---|
| `sym` | `qmax = 2^(b-1) - 1`, `s = max|w| / qmax`, `q = clamp(round(w/s), -qmax, qmax)`, `w_hat = q*s` |
| `asym` | `lo = min(min w, 0)`, `hi = max(max w, 0)`, `s = (hi - lo) / (2^b - 1)`, `z = round(-lo/s)`, `q = clamp(round(w/s) + z, 0, 2^b - 1)`, `w_hat = (q - z)*s` |

Notes:

- `sym` uses the restricted, zero-centred range, so it has `2^b - 1` levels.
  At 2 bits that is {-s, 0, +s}.
- `asym` widens the range to include 0, so zero is always exactly representable.
- A group with zero range (for example, all zeros) gets `s = 1`. No division by
  zero occurs, and an all-zero group round-trips exactly.
- Every element satisfies `|w - w_hat| <= s/2` (tested).

**Fake quantization** ([`src/qtk/model.py`](src/qtk/model.py)): `apply_fake_quant`
replaces the weight of **every `nn.Linear`** with `dequantize(quantize(W))`. For
Pythia that is the attention QKV and output projections, both MLP matrices, and
the untied LM head (`embed_out`). Biases, LayerNorms and the input embedding
(`nn.Embedding`) stay at full precision. Each bit-width is quantized from a saved
copy of the original weights, so errors never compound across a sweep, and
`restore_weights` puts the originals back bit-exactly.

## Experimental defaults

**Precision.** The **FP32 model is the reference**. Quantized models are
fake-quantized in FP32 *and run in FP32*, so the only difference from the
reference is the quantization itself. An **FP16** run of the unquantized model
is reported as an extra baseline row on CUDA. It is skipped on CPU, where
half-precision kernels are slow and unrepresentative.

**Default quantizer for Exp 2 and 3:** asymmetric, group size 128
(`--scheme asym --granularity group:128`). Exp 1 sweeps every combination of
{sym, asym} × {tensor, channel, g32, g64, g128} at 2–8 bits.

**Models:** `EleutherAI/pythia-{70m,160m,410m}` at revision `step143000`, the
final training checkpoint (the same weights as `main`), pinned for reproducibility.

**Exp 1 (reconstruction error):**
- Uses all `nn.Linear` weights of pythia-70m and pythia-160m.
- Per layer: MSE and relative error `||W - W_hat||_F / ||W||_F`.
- The plot shows the pooled relative error `sqrt(sum ||W - W_hat||^2 / sum ||W||^2)`.
- Outputs `error.csv` (per layer), `error_summary.csv` (also has the mean of per-layer errors) and `error.png`.

**Exp 2 (perplexity):**
- Uses the WikiText-2 raw test split (`Salesforce/wikitext`, `wikitext-2-raw-v1`).
  Lines are joined with `"\n\n"`, tokenized once, and cut into
  **1024-token non-overlapping chunks**; the trailing partial chunk is dropped.
- PPL = exp(mean next-token NLL over all predicted tokens). The first token of
  each chunk has no context and is not scored. Log-softmax is computed in FP32.
- `--max-chunks N` keeps the first N chunks.
- Outputs `ppl.csv` and `ppl.png`.

**Exp 3 (in-context learning, SST-2):**
- **Eval set:** `stanfordnlp/sst2` validation (872 examples). A fixed
  **500-example** subset (`--n-eval`) is chosen with seed `--seed`
  (default 0), stratified by label.
- **Prompt:** `Review: {sentence}\nSentiment: {label}\n\n` repeated k times,
  then `Review: {query}\nSentiment:`.
- **Scoring:** the prediction is whichever of the label words **" negative" /
  " positive"** has the higher summed log-likelihood as the continuation. The
  CSV also reports the mean log-likelihood margin of the correct label.
- **Demos** come from the SST-2 train split. Sentences shorter than 5 words are
  excluded, because train contains many short phrase fragments.
  - Each eval example gets one 16-demo sequence per seed, built from
    (negative, positive) pairs in random order within each pair. The k-shot prompt
    uses the first k demos, so every even k is exactly label-balanced, and k = 1
    gets a random label.
  - The demos depend only on (seed, example), so **every precision sees exactly
    the same prompts**, and comparisons across precisions are paired.
- **Seeds:** 3 demo seeds by default (`--seeds "0,1,2"`); use 5 for the final
  run. k = 0 does not depend on the seed, so it is evaluated once and reported
  under each seed.
- **Long prompts:** prompts longer than the context window (2048 tokens) would be
  truncated from the left. The `n_truncated` column counts this; it should be 0
  for SST-2 at k ≤ 16.
- **Outputs:**
  - `icl.csv` (one row per model × precision × k × seed)
  - `icl_summary.csv` (mean ± std over seeds, and `gain` = acc(k) − acc(0))
  - `icl.png` (accuracy vs k)
  - `icl_gain.png` (gain at the largest k vs precision)
  - `--save-preds` also writes per-example log-probs to `icl_predictions.csv`,
    for paired significance tests.

## Reproducibility

- `set_seed` seeds Python, NumPy and PyTorch, and turns on
  `torch.use_deterministic_algorithms` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- It also disables TF32 (`torch.set_float32_matmul_precision("highest")`), so
  FP32 runs really are IEEE float32 on the GPU.
- Quantization is deterministic (round-to-nearest), and evaluation batching is
  deterministic.
- Repeating a run on the same machine and software stack gives byte-identical
  CSVs; this is tested. Results can differ in the last digits between CPU and
  GPU, or across library versions.
- `*_meta.json` records the exact versions and device used.

## Results

Not yet filled in: this section gets the numbers and figures from the final GPU run.

## Limitations

- **Fake quantization only.** Weights are quantized then dequantized, which
  measures accuracy loss but gives no memory or speed benefit.
- **Round-to-nearest only.** There are no calibration-based methods (GPTQ, AWQ),
  no outlier handling and no activation quantization. This is deliberate: the
  prediction under test is about plain weight-only rounding.
- **Embeddings, biases and LayerNorms stay full precision.** The LM head *is*
  quantized.
- **Small scale.** The models are small (70m–410m), and SST-2 is a simple binary
  task, so absolute few-shot accuracies are modest. The comparison that matters is
  across precisions for the same model.

## Layout

```
src/qtk/quant.py       quantizers (QuantConfig, quantize, dequantize, fake_quant)
src/qtk/model.py       model loading, apply_fake_quant / restore_weights
src/qtk/data.py        WikiText-2 chunks, SST-2 subset and demo sampling
src/qtk/exp_error.py   Exp 1
src/qtk/exp_ppl.py     Exp 2
src/qtk/exp_icl.py     Exp 3 (prompting, label scoring)
src/qtk/plotting.py    figures
src/qtk/cli.py         Typer CLI (qtk)
tests/                 pytest: quantizer properties, model wrapping, ICL scoring, offline CLI runs
```
