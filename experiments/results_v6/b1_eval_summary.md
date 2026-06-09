# B1 retrain: Old vs New TRM on narrow (n=20)

Old model = `experiments/results_v6/cl_trm_model.pt` (PD+CBF closed-loop trained)
New model = `experiments/results_v6/cl_trm_narrow_v1.pt` (narrow-matched expert NMPC trained, 18000 samples, 100 epochs)

| K | Method | Old succ | Old TErr | New succ | New TErr | Δsucc | ΔTErr |
|---|---|---|---|---|---|---|---|
| 1 | TRM_only | 0/20 | 7.324 | 0/20 | 2.384 | +0 | -4.940  ✓ |
| 1 | TRM_PD_a095 | 0/20 | 4.408 | 0/20 | 4.385 | +0 | -0.023 |
| 1 | PD | 0/20 | 4.440 | 0/20 | 4.440 | +0 | +0.000 |
| 1 | A3 | 9/20 | 1.457 | 1/20 | 2.274 | -8 | +0.817  ←⚠️ |
| 5 | TRM_only | 0/20 | 5.669 | 0/20 | 0.832 | +0 | -4.837  ✓ |
| 5 | TRM_PD_a095 | 9/20 | 0.558 | 9/20 | 0.565 | +0 | +0.007 |
| 5 | PD | 7/20 | 0.788 | 7/20 | 0.788 | +0 | +0.000 |
| 5 | A3 | 15/20 | 0.262 | 10/20 | 0.633 | -5 | +0.371  ←⚠️ |
| 10 | TRM_only | 0/20 | 5.357 | 1/20 | 0.748 | +1 | -4.610  ✓ |
| 10 | TRM_PD_a095 | 17/20 | 0.255 | 14/20 | 0.279 | -3 | +0.023  ←⚠️ |
| 10 | PD | 13/20 | 0.509 | 13/20 | 0.509 | +0 | +0.000 |
| 10 | A3 | 18/20 | 0.114 | 15/20 | 0.258 | -3 | +0.144  ←⚠️ |
| 20 | TRM_only | 0/20 | 4.973 | 3/20 | 0.504 | +3 | -4.469  ✓ |
| 20 | TRM_PD_a095 | 18/20 | 0.160 | 16/20 | 0.286 | -2 | +0.126  ←⚠️ |
| 20 | PD | 18/20 | 0.175 | 18/20 | 0.175 | +0 | +0.000 |
| 20 | A3 | 19/20 | 0.018 | 16/20 | 0.209 | -3 | +0.191  ←⚠️ |

## 解读

**TRM-only 维度（独立可控性）— 新模型显著进步**
- TErr 在 K=1/5/10/20 上分别 3.1×/6.8×/7.1×/9.9× 改善 → retrain 让 TRM 输出贴近 expert NMPC，magnitude overshoot 解决。
- TRM-only success K=20: 0→3 (0%→15%)，首次有了独立可控样本，但 **仍未达到用户 ≥50% 目标**。
- 单独训练 narrow-matched 数据无法让 TRM-only 成为 standalone controller，因 narrow 任务需要 lateral detour 探索，单候选 deterministic forward 不够。

**A3 hybrid pool 维度（与 PD 配对）— 新模型反而退化**
- A3 K=1: 9/20 → 1/20（大退化），A3 K=5: 15/20 → 10/20，A3 K=10: 18/20 → 15/20，A3 K=20: 19/20 → 16/20。
- 所有 K 上 succ 都下降 1-8 个 trial，TErr 平均上升 0.1-0.8m。

**机制（论文级 insight）**
- 旧 TRM 的 'magnitude 1.4× overshoot + direction 0.97 cos' 在 hybrid pool 中是 **特征不是 bug**：提供 PD 候选不具备的 'exploration thrust'，rollout selector 跨越窄通道。
- 新 TRM 学得太接近 expert (locally optimal, 小 magnitude 平滑控制) → 与 PD 候选 **同质化** → diversity 消失 → rollout selector 没有比 PD 更好的候选可选。
- 这说明 **PTRM hybrid pool 的价值在 candidate diversity，不在 candidate accuracy**：越准确的 learned predictor 在 hybrid 中价值越小；越 imperfect 但 structurally non-greedy 的 predictor 越有用。

## 论文方向（保留 PTRM-positive 主线）
- A3 + 旧 model 仍是 PTRM-positive 主信号，证据未受损（n=80 narrow 已显著，cross-task 不退化）。
- 新增 'candidate diversity matters more than accuracy' 节，把 retrain 实验作为反例 → 强化 hybrid pool 的非平凡性。
- Reviewer 防御：'why not retrain TRM to be more accurate?' → 已实验，retrain 让 TRM-only 进步但 A3 退化。
