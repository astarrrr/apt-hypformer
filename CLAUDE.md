# CLAUDE.md

## Project Overview

This project integrates a **Hyperbolic Transformer** encoder into the **PIDSMaker** provenance-graph intrusion detection pipeline. It replaces PIDSMaker's standard GNN encoder with a dual-branch architecture (Hyperbolic Transformer + GNN) that operates in Lorentz hyperbolic space, using edge reconstruction error as anomaly scores.

- **Upstream dependency**: PIDSMaker at `/home/astar/projects/PIDSMaker/`
- **Adaptation source**: HypFormer at `/home/astar/projects/hyperbolicTransformer-master/large/`
- **Runtime dependency**: `geoopt>=0.5.0`

## Architecture

```
Euclidean input [N, in_dim]
        |
   +----+----+
   |         |
TransConv  GraphConv
(Lorentz)  (Euclidean)
   |         |
   |    HypLinear(euc->hyp)
   |         |
   +---mid_point---+   (weighted Lorentz midpoint fusion)
         |
    HypLinear (output projection)
         |
  Lorentz output [N, out_dim+1]
         |
    gather_h (PIDSMaker)
         |
  HypEdgeDecoder (logmap0 -> MLP -> expmap0)
         |
  HypEdgeReconstruction (Lorentz distance loss)
```

## Directory Structure

```
myproject/
├── CLAUDE.md
├── manifolds/              # Lorentz manifold math (adapted from HypFormer)
│   ├── utils.py            # Numerical stability: LeakyClamp, acosh, sqrt, clamp
│   ├── lorentz_math.py     # Low-level ops: inner, expmap0, logmap0, parallel_transport
│   ├── lorentz.py          # Lorentz class (extends geoopt.Lorentz): mid_point, square_dist
│   └── layers.py           # HypLinear, HypLayerNorm, HypActivation, HypDropout
├── encoders/
│   └── hyp_transformer_encoder.py  # Core: HypTransformerEncoder (dual-branch)
├── decoders/
│   └── hyp_edge_decoder.py         # HypEdgeDecoder (tangent-space MLP)
├── objectives/
│   └── hyp_edge_reconstruction.py  # HypEdgeReconstruction (Lorentz midpoint target)
├── losses/
│   └── hyp_losses.py               # lorentz_distance_loss, tangent_mse_loss
├── optimizer.py            # DualOptimizer (Adam + RiemannianAdam)
├── factory_ext.py          # Factory functions bridging PIDSMaker <-> this project
└── configs/
    └── hyp_pids.yml        # Full pipeline YAML config
```

## Key Conventions

### Dimension Convention
- `node_out_dim` in config = **spatial** dimension (e.g., 32)
- Actual tensor shape = `node_out_dim + 1` (e.g., 33) because Lorentz points have a time component at index 0
- All modules follow this: encoder outputs `[N, out_dim+1]`, decoder expects `[E, dim+1]`

### Encoder Output Format
The encoder returns `{"h": tensor}` where `h` has shape `[N, out_dim+1]`. PIDSMaker's `Model.gather_h()` indexes with `h[edge_index[0]]` / `h[edge_index[1]]` which works directly on Lorentz tensors.

### Objective Interface
Must match PIDSMaker's pattern:
```python
def forward(self, h_src, h_dst, inference=False, **kwargs) -> dict:
    # Training:  return {"loss": scalar}
    # Inference: return {"loss": per_edge_tensor [E,]}
```

### Euclidean-to-Lorentz Mapping
`HypLinear` with `x_manifold='euc'` prepends a ones column and applies `expmap0` before the linear transform. This is the standard entry point from Euclidean to Lorentz space.

## Modified PIDSMaker Files

Only 2 files in PIDSMaker are modified (4 insertion points total):

### `pidsmaker/factory.py`
1. **Line 8-14**: Import `sys` and `factory_ext` functions
2. **Line 238-239**: `encoder_factory()` — added `hyperbolic_transformer` branch
3. **Line 392-395**: `objective_factory()` — intercept `hyperbolic_edge_reconstruction` before standard decoder matching (uses `continue` to skip `decoder_matches_objective` check)
4. **Line 691-692**: `optimizer_factory()` — conditional `DualOptimizer` when encoder is `hyperbolic_transformer`

### `pidsmaker/config/config.py`
1. `ENCODERS_CFG`: added `"hyperbolic_transformer"` with 11 hyperparameters
2. `OBJECTIVES_EDGE_LEVEL`: added `"hyperbolic_edge_reconstruction"`
3. `TASK_ARGS["training"]`: added `hyp_lr` and `hyp_weight_decay`

## Full-Graph Strategy

Set `time_window_size: 99999999` so PIDSMaker's graph builder places all events into a single window, producing the complete provenance graph. PIDSMaker's `intra_graph_batching: edges` mode splits this large graph into batches of `intra_graph_batch_size` edges. The encoder runs on each batch:
- GNN branch uses the batch's local `edge_index`
- Transformer branch applies global attention over the batch's nodes
- `linear_focused` attention has O(n) complexity, making this feasible for large batches

## Config Reference (`configs/hyp_pids.yml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `intra_graph_batch_size` | 4096 | Edges per batch (reduce if OOM) |
| `trans_num_layers` | 2 | Transformer depth |
| `trans_num_heads` | 4 | Attention heads |
| `gnn_num_layers` | 2 | GNN depth |
| `graph_weight` | 0.5 | GNN fusion weight (0=pure Transformer, 1=pure GNN) |
| `k` | 1.0 | Lorentz curvature parameter (curvature = -1/k) |
| `attention_type` | `linear_focused` | `linear_focused` (O(n)) or `full` (O(n^2)) |
| `power_k` | 2 | Power for linear focused attention kernel |
| `hyp_lr` | 0.0001 | Learning rate for ManifoldParameters |
| `node_out_dim` | 32 | Spatial embedding dim (tensor is 33-dim) |

## Common Tasks

### Run the pipeline
```bash
cd /home/astar/projects/PIDSMaker
python main.py --dataset CADETS_E3 --config /home/astar/projects/myproject/configs/hyp_pids.yml
```

### Quick smoke test (Python)
```python
import sys; sys.path.insert(0, '/home/astar/projects/myproject')
import torch
from encoders import HypTransformerEncoder

enc = HypTransformerEncoder(in_dim=128, hid_dim=64, out_dim=32,
    trans_num_layers=2, trans_num_heads=4, trans_dropout=0.3,
    gnn_num_layers=2, gnn_dropout=0.3, graph_weight=0.5,
    k=1.0, attention_type='linear_focused', power_k=2,
    use_bn=True, use_residual=True)
enc.eval()
with torch.no_grad():
    out = enc(torch.randn(50, 128), torch.randint(0, 50, (2, 100)))
assert out["h"].shape == (50, 33)  # 32 spatial + 1 time
```

## Numerical Stability Notes

- `manifolds/utils.py` provides `LeakyClamp` (gradient-preserving clamp), stable `acosh`, and `sqrt` with `min=1e-9`
- `lorentz_math.py` clamps `expmap` norm to `EXP_MAX_NORM=10` to prevent overflow
- Division denominators are clamped to `1e-7` or `1e-8` throughout
- BatchNorm layers in GNN branch skip when `N <= 1` to avoid errors on tiny batches
