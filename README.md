# GroupPatent with Feature Memory

Minimal source release of the feature-memory patent-generation method. Requires Python 3.10+.

## Included

- Draft-derived technical feature extraction and caching.
- Feature context injection, coverage tracking, and incremental repair.
- Shared three-layer generation framework and prompt templates.
- Offline smoke checks and patent-format quality tests.

`agent_v2` is a runtime dependency of `FeatureMemoryPipeline`, not a separately exposed baseline method. Original module paths and generation logic are retained. The public entry point exposes only `memory_only`, `memory_tracking`, and `full_repair` (default).

Excluded: baseline runners, agent_v1, zero-shot/sequential entry points, comparison protocols, real datasets, experimental outputs, manual review sheets, paper notes, credentials, caches, and Git history.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-own-key"
export OPENAI_API_ENDPOINT="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen-plus"
```

`.env.example` documents the variables; .env files are not automatically loaded. Choose an endpoint and model supported by your provider. The default endpoint belongs to DashScope. Generation incurs API charges.

## Run

```bash
python run_feature_memory.py --input examples/draft.json --output-dir outputs
python run_feature_memory.py --input /path/to/json-directory --output-dir outputs/batch
```

Every UTF-8 input JSON must contain a nonempty `draft` object or string; `metadata` is optional. Reference patents are not required or passed to generation. The example is synthetic, not evaluation data. Output contains `patent`, `feature_memory`, and the effective configuration. Existing output files are not overwritten. Runs are sequential because the shared generation memory is process-wide.

Optional flags: `--mode memory_only|memory_tracking|full_repair`, `--feature-model MODEL`, `--coverage-model MODEL`, `--coverage-mode llm|hybrid|lexical`. Hybrid coverage can fall back to lexical matching. Use an independently chosen semantic evaluator for rigorous studies. Extracted features are model-derived candidates, not human-validated findings. This release does not reproduce excluded baseline comparisons.

## Offline Checks

```bash
python -m experiments.feature_memory_experiment.smoke_test
python -m unittest discover -s tests -v
python run_feature_memory.py --help
```

These checks do not call an API. End-to-end generation requires credentials and a reachable provider. Generated patent text requires qualified review.

## Publication Notes

`SOURCE_MANIFEST.json` records SHA-256 hashes of packaged files. Do not upload local keys, data, or generated results. No license is assigned by this packaging step: the owner must choose a license and add authorship/citation information before granting reuse rights.
