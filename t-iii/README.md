# T-III: video-grounded reward generation

This folder converts one empirically identified Push-T failure from T-II into:

1. a dense reward function operating on ManiSkill state; and
2. an initial-state sampler biased toward that failure regime.

The implementation is one CLI with three modules:

- `ripl_t3/pipeline.py`: video grounding, reward planning, code generation, and artifact saving;
- `ripl_t3/state_api.py`: the small ManiSkill state interface exposed to generated rewards; and
- `ripl_t3/validation.py`: static and numerical checks for generated code.

## Reproduce

Python 3.10-3.12 is supported.

```bash
cd t-iii
python -m pip install -e '.[dev,llm]'
python -m pytest -q
```

Create a failure configuration from the final T-II pose bounds. The file must contain a
`failure_mode_name`, `analyst_description`, and `pose_region`; see
`configs/failure_mode.example.json`. The example is a placeholder, not an experimental result.

Then run one representative video with its exact T-II episode record:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-vllm
export RIPL_T3_MODEL=Qwen/Qwen3.8-27B

python generate_t3.py \
  --video /path/to/mode_1/videos/0.mp4 \
  --failure-config configs/failure_mode_1.json \
  --episode-record /path/to/mode_1/episodes.jsonl \
  --episode-index 0 \
  --output-dir artifacts/mode_1/candidate_000
```

Use `--dry-run` to inspect the video-grounding prompt without calling an LLM. Use
`--recorded-responses responses.json` to replay a saved generation without an API call.

## Outputs

Each new output directory contains six core files:

- `reward.py` and `episode_sampler.py`: generated executable code;
- `prompts.json` and `responses.json`: the complete LLM interaction;
- `manifest.json`: model, timestamp, video hash, T-II evidence, and pose range;
- `validation.json`: static and numerical checks.

These small artifacts should be committed. T-IV should add its training configuration and metrics
beside them; large checkpoints and videos should use Git LFS or a release rather than ordinary Git.

The generator refuses to overwrite an existing output directory. Record the model revision, API or
vLLM version, GPU, seeds, and T-IV training command alongside the artifact before submission.

## Research rationale

The three stages are intentionally visible in `pipeline.py`:

1. **Grounding:** separate visual evidence from uncertain causal claims.
2. **Planning:** turn the observed failure into staged, interpretable reward terms and a mixed reset
   distribution.
3. **Coding:** generate the reward and sampler against an explicit batched state API.

This follows [ROSETTA](https://sanjanasrivastava.github.io/rosetta-project/), which separates
grounding, staging, and coding. It also follows
[Eureka](https://eureka-research.github.io/): generated reward code is a candidate that must be
evaluated using an unchanged ground-truth task metric, then refined using training feedback.

Validation checks syntax, dangerous capabilities, tensor shapes, finite values, preference for a
successful aligned state, reset-domain bounds, failure-region sampling mass, and retained global
exploration. These checks are screening tools, not proof that a reward is useful or safe. Manually
review accepted code and test for reward hacking, oscillatory progress bonuses, proxy optimization,
and poor RL learnability.

## Model and GPU

`Qwen/Qwen3.8-27B` natively accepts video and its unquantized repository is approximately 55.6 GB.
Use one 80+ GB GPU for the reproducible path; the existing 48 GB A40 is not a safe unquantized
target. `scripts/serve_qwen_vllm.sh` verifies VRAM and launches a bounded-context vLLM server. A
hosted multimodal API can be substituted by changing only the model, base URL, and API key.

## T-IV interface

T-IV should import `pusht_state_from_env` from `ripl_t3.state_api`, load the generated
`compute_dense_reward` and `sample_initial_poses` functions, and train with fixed RL hyperparameters.
Its README/result table should report the base policy and improved policy on the same T-II targeted
episodes and seeds. Do not select a reward by training return alone; retain `success_once` as the
primary metric.
