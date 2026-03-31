# BEST-Route — External Baseline

## Paper

**BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute**
Ding, Mallick, Zhang, Wang, Madrigal, Garcia, Xia, Lakshmanan, Wu, Rühle
*Forty-second International Conference on Machine Learning (ICML 2025)*
arXiv: <https://arxiv.org/abs/2506.22716>

```bibtex
@inproceedings{dingbest,
  title={BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute},
  author={Ding, Dujian and Mallick, Ankur and Zhang, Shaokun and Wang, Chi
          and Madrigal, Daniel and Garcia, Mirian Del Carmen Hipolito
          and Xia, Menglin and Lakshmanan, Laks VS and Wu, Qingyun
          and R{\"u}hle, Victor},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

## Official Code

<https://github.com/microsoft/best-route-llm>  (MIT license)

## Status

**Blocked** — the official code requires a multi-stage pipeline that cannot
be run from this repository without new API calls and external model
downloads.  See `docs/BEST_ROUTE_INTEGRATION.md` and
`docs/BEST_ROUTE_INTEGRATION_STATUS.md` for details.

Two classes exist in `src/baselines/external/best_route_wrapper.py`:

| Class | Status | Description |
|-------|--------|-------------|
| `BESTRouteBaseline` | Blocked | Official-code wrapper; requires `.repo` below |
| `BESTRouteAdaptedBaseline` | ✅ Runnable | Compatibility adaptation; no external deps |

## Setup for Official Code (Optional)

If you want to run the full official BEST-Route pipeline:

1. Clone the official repository:
   ```bash
   cd external/best_route
   git clone https://github.com/microsoft/best-route-llm .repo
   ```

2. Install requirements:
   ```bash
   cd .repo
   pip install -r requirements.txt
   pip install -r notebooks/requirements_data_preparation.txt
   ```

3. Complete the data preparation pipeline as described in `.repo/README.md`:
   - Dataset mixing (10 000 prompts)
   - Response sampling (20 samples/query/model for each candidate LLM)
   - armoRM oracle reward scoring
   - Proxy reward model training (DeBERTa fine-tuning)
   - Router training (DeBERTa-v3-small, `prob_nlabels` loss)

4. Once the router is trained, implement the bridge in
   `src/baselines/external/best_route_wrapper.py → BESTRouteBaseline.solve()`.

**Note**: Steps 3 and 4 require significant compute, multiple LLM API keys,
and are not compatible with this repository's single-model binary-routing
setting without further adaptation.

## Running the Compatibility Adaptation

The `BESTRouteAdaptedBaseline` is immediately runnable:

```python
from src.baselines.external.best_route_wrapper import BESTRouteAdaptedBaseline
from src.models.dummy import DummyModel

model = DummyModel(correct_prob=0.6, seed=42)
model.set_ground_truth("42")
baseline = BESTRouteAdaptedBaseline(model, threshold=0.5)
result = baseline.solve("q1", "What is 6 × 7?", "42", n_samples=2)
print(result.action)
```

For paper-grade evaluation, substitute `DummyModel` with the real LLM backend
and run `scripts/run_strong_baselines.py` with
`configs/best_route_adapted.yaml`.

