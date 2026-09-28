# Local setup notes

Upstream repo: https://github.com/ml-postech/RA-RAG (EMNLP 2025, Hwang et al.)
Paper: arXiv:2410.22954

## Environment

- Python 3.11 venv at `.venv` (the system default was 3.14, which has no prebuilt
  wheels yet for torch/numpy/faiss — installing with it forces slow, failure-prone
  source builds). Activate with `source .venv/bin/activate`.
- `requirements.txt` is the upstream file, unmodified.
- `requirements-lock.txt` is what's actually installed and known to work on this
  machine (macOS arm64, Python 3.11). Use `pip install -r requirements-lock.txt`
  to reproduce this environment exactly.
- `pip install -r requirements.txt` alone will fail on this machine: `beir` pulls
  in `datasets==1.1.1`, which predates pyarrow's current API and breaks on import
  (`AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`). Fixed by
  upgrading `datasets>=2.14` after the base install (already reflected in the lockfile).
  RA-RAG's own code only imports `beir.*` directly, not `datasets`, so this upgrade
  doesn't touch anything the repo relies on.
- spaCy model: `python -m spacy download en_core_web_sm` (already run).

## Added on top of upstream

Upstream only supports local HF models (see `model_configs/`, e.g. Llama-3-8B-Instruct,
Phi-3-mini) loaded via `transformers`. We're running a hybrid setup instead:

- **Retrieval / vector index:** unchanged — FAISS, local, in-process (via `faiss-cpu`
  and BEIR's `DenseRetrievalExactSearch`). No hosted vector DB.
- **Embeddings:** unchanged — the repo's own contriever pipeline
  (`src/contriever_src/`), CPU-friendly for the corpus sizes this thesis uses.
- **LLM (the part that changes):** Gemini API instead of local Llama-3-8B/Phi-3-mini,
  to avoid needing a GPU. Added `google-genai` + `python-dotenv`.

Put your key in `.env` (copy `.env.example`, git-ignored):

```
GEMINI_API_KEY=your-key-here
```

### Where the swap happens

`src/utils.py` now branches inline instead of introducing a shared backend
abstraction — same two functions upstream already had, extended with a Gemini
path, no new classes:

- `load_models(args)` (`src/utils.py:47`): if `model_configs/<name>_config.json`
  has `"provider": "gemini"`, returns a plain `dict` (`{provider, client,
  model_name}`) wrapping a `genai.Client` instead of loading
  `AutoModelForCausalLM` + building an HF `pipeline`. Reads `GEMINI_API_KEY` from
  `.env` via `python-dotenv` (or `model_config["api_key"]` if you'd rather set it
  per-config).
- `inference(pipeline, ...)` (`src/utils.py:302`): checks
  `isinstance(pipeline, dict) and pipeline.get('provider') == 'gemini'` at the top;
  if so, calls `gemini_generate_content(...)` per query and wraps the result as
  `[{'generated_text': text}]` — the exact shape the HF branch already produced —
  so `extract_final_answer(..., prompt_id=4)` and everything downstream in
  `ra_rag_utils.py` needs zero changes. Falls through to the original
  `transformers.pipeline(...)` call otherwise.
- `gemini_generate_content(...)` (`src/utils.py:71`): the actual API call, with a
  5-attempt exponential backoff retry (transient network/rate-limit errors only).

Use it by passing `--model_name gemini` on the CLI (`model_configs/gemini_config.json`
is the new config, model `gemini-2.5-flash` by default) instead of the upstream
`--model_name llama3_8b` / `phi3_mini`.

`src/reliability_utils.py:173` (`QAConvertor`, the BART-based declarative-answer
converter used for filtering) is untouched — it's a small local model unrelated to
which backend answers the source queries, but it hardcodes `.to('cuda')`, which
will fail on this Mac (no CUDA) if that filtering path is exercised. Not fixed yet;
flag it if `main.py` errors there.

## Interactive webapp (`webapp/`)

A small local playground that follows the paper's actual two-phase structure —
not a shortcut of it — for two things: running RA-RAG normally, and manually
driving the build-then-betray attack.

It does **not** go through `main.py` / BEIR / FAISS / contriever — those exist to
retrieve context automatically from a corpus, which isn't the point of an
interactive demo. Instead you paste each source's context directly. But the
*mechanism* mirrors `main.py` exactly:

1. **Estimation phase (calibration rounds).** Each round's per-source answers
   accumulate; `iterative_weighted_majority_voting`
   (`src/reliability_utils.py`, unmodified) gives a live reliability preview,
   same as `multi_source_weight_est`.
2. **Lock.** Weights freeze — same as `main.py` computing `estimated_weight_lst`
   once and handing it to `reliablity_aware_inference` rather than re-estimating
   per query.
3. **Inference phase.** Each query only consults the **top-κ most reliable
   frozen-weight sources** (κ-RRSS, `top_k_src`) — lower-weight sources are
   excluded from that round entirely, not just down-weighted. The chosen
   sources' answers are run through the paper's own semantic clustering
   (`EntailmentDeberta` + `get_semantic_ids` + `unify_answer`) so paraphrases
   don't split the vote, then `weighted_majority_voting` decides the consensus
   using the frozen weights.

Answer generation uses `wrap_prompt` (`src/prompts.py`) and
`gemini_generate_content` (`src/utils.py`) per source, same as before.

**AlignScore grounding filter (calibration only, matches the paper):** every
calibration-round answer — Gemini-generated or manually forced — is converted
to a declarative claim (`QAConvertor`, the repo's BART model) and scored
against that source's own context for the round (`AlignScore-base`, the repo's
vendored `Alignscore/` package). Below `ALIGN_SCORE_THRESHOLD` (env var,
default `0.5`, matching `main.py`'s intended default), the answer is replaced
with `"i don't know"` before it ever reaches the reliability estimator —
exactly what `convert_to_valid_answer` does inside `multi_source_weight_est`.
Never applied during inference, since the paper doesn't either
(`reliablity_aware_inference` has no filtering step). The webapp shows a
"filtered by AlignScore" badge on any calibration answer this replaces.

Needs the checkpoint the repo doesn't ship (gitignored, not fetched by
`requirements.txt`):
```
curl -L -o Alignscore/ckpt_dir/AlignScore-base.ckpt \
  https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt
```
~1.9GB, official checkpoint from the AlignScore paper's own author (`yzha`,
ACL 2023). `engine.get_align_scorer()` raises a clear error naming this command
if the file is missing.

**Device note:** `EntailmentDeberta`, `QAConvertor`, and `AlignScoreFiltering`
all hardcoded `cuda`/`cuda:0` in the original repo — patched in
`src/reliability_utils.py` to auto-detect `cuda` → `mps` → `cpu` (`DEVICE`
constant near the top of the file); `Alignscore/inference.py`'s
`load_from_checkpoint` also needed `map_location="cpu"` added, since the
checkpoint was saved with CUDA tensors and would otherwise fail to
deserialize on a machine with no CUDA. First use of either model downloads +
loads its weights (~1.5GB DeBERTa, ~1.9GB AlignScore) and the AlignScore
checkpoint load alone took ~5 minutes one-time on this machine (MPS); every
call after that, in the same server process, is sub-second since both models
stay resident in memory. Restarting the server pays the load cost again.

Run it:

```
source .venv/bin/activate
uvicorn webapp.backend.main:app --reload --port 8000
```

Then open http://localhost:8000. Needs `GEMINI_API_KEY` in `.env` for any source
whose answer isn't manually forced.

- **Normal tab:** create a session, run a few calibration rounds (pasting
  context per source each time), **lock weights**, then run inference rounds —
  only the top-κ sources you configured get consulted per query, and answers go
  through semantic clustering before voting.
- **Attack tab:** one or more adversarial source names (comma-separated — several
  means colluding sources that all get the same scripted answer each round).
  During calibration, the adversary is "honest" (real Gemini call) or "mimic"
  (directly supply the known-correct answer, cheaply building trust). After
  locking, during inference, it's "honest" or "payload" (force a wrong answer).
  The banner distinguishes three outcomes: the adversary wasn't even in the
  top-κ consulted this round (excluded by the defense before voting even
  happened), it was consulted but got outvoted, or it was consulted and won.

A lone adversary's weight is capped at 1.0 (it's literally an accuracy
fraction) and can never outvote two honest sources that agree with each other —
so a genuinely flipping attack needs either multiple colluding sources or
honest sources with imperfect reliability of their own. The attack-tab presets
demonstrate both failure and success cases, plus the κ-RRSS defense actually
excluding a tied-weight adversary when κ is lowered.

**The fourth preset is the more interesting single-adversary path** (H2
material): a lone, non-colluding adversary that never has to know a true
answer at all. With 3 honest sources genuinely disagreeing with each other on
disputed trivia (real historical misattributions — Bell/Meucci, Edison/Swan,
Amundsen/Scott — each one plausible, not arbitrary noise), the adversary just
sides with whichever pair of honest sources already agrees. That's free: it
banks a perfect 1.0 track record without ever independently verifying
anything, while each individual honest source ends up discounted to 0.75
(wrong exactly once, by disagreeing with a peer). Once every *individual*
honest source is weaker than the adversary, top-κ selection — the mechanism
meant to filter out unreliable sources — actually helps the attacker: lowering
κ excludes the *other* honest sources that would have outvoted it, leaving a
1-on-1 matchup the adversary wins outright. Verified end-to-end via direct API
calls (weights land at exactly `[0.75, 0.75, 0.75, 1.0]`, betrayal round
consensus flips to the payload with only `[Adversary, <one honest source>]`
consulted).

This preset scripts every source's answer explicitly (`honestAnswers` in the
preset data, a preset-only parameter — the interactive UI never sets it) rather
than relying on live Gemini calls for the honest sources. Calibration has no
semantic clustering (only inference does), so exact-string wording differences
between Gemini calls for the same fact ("edison" vs "thomas edison") would
silently break a demo built around an exact agreement/disagreement pattern.
The interactive UI still uses real Gemini calls throughout; only this one
preset trades that for determinism.

**Bug found and fixed while building this:** `update_reliability`
(`src/reliability_utils.py`) divides by each source's answer count with no
zero-guard — if every one of a source's answers gets filtered to "i don't
know" (e.g. everything supplied with no grounding context), that's 0/0, which
produced a `NaN` that then crashed the API response (`ValueError: Out of range
float values are not JSON compliant`). Patched to fall back to a neutral 0.5
prior when a source has zero counted answers, instead of propagating NaN. This
is a latent gap in the paper's own code, not something introduced by AlignScore
filtering — it just never surfaces in the original pipeline since real
Gemini/local-LLM answers with real contexts rarely go fully ungrounded.

State is in-memory only (`webapp/backend/engine.py`'s `SESSIONS` dict) —
restarting the server clears all sessions, and the loaded DeBERTa model.
Intentional; this is a scratch tool for poking at the algorithm, not a place to
store real experiment results (those still come from `main.py` + the shell
scripts, saved to `results/`).
