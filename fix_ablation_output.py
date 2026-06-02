# -*- coding: utf-8 -*-
"""
从已保存的消融实验数据恢复并重新生成 JSON 和图表
"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'experiments', 'results')

# 加载消融实验数据
npy_file = os.path.join(RESULTS_DIR, 'ablation_experiments_20260602_050252_metrics.npy')
data = np.load(npy_file, allow_pickle=True).item()

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

# 重新保存 JSON
json_path = os.path.join(RESULTS_DIR, 'ablation_experiments_fixed.json')
with open(json_path, 'w') as f:
    json.dump(make_serializable(data), f, indent=2)
print(f"消融实验 JSON 已保存至: {json_path}")

# 打印消融实验结果摘要
for ablation_name, ablation_data in data.items():
    print(f"\n--- {ablation_name} ---")
    for label, metrics in ablation_data.items():
        print(f"  {label}: Success={metrics['success_rate']:.1f}%, Pos IAE={metrics['p_iae_mean']:.2f}±{metrics['p_iae_std']:.2f}, Collision={metrics['collision_count']}")

# 绘制消融实验图
num_trials = 200

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# K 消融
k_data = data['K']
k_labels = list(k_data.keys())
k_success = [k_data[l]['success_rate'] for l in k_labels]
k_p_iae = [k_data[l]['p_iae_mean'] for l in k_labels]
axs[0, 0].bar(range(len(k_labels)), k_success, color='steelblue', alpha=0.8)
axs[0, 0].set_xticks(range(len(k_labels)))
axs[0, 0].set_xticklabels(k_labels, rotation=45)
axs[0, 0].set_ylabel('Success Rate (%)')
axs[0, 0].set_title('(a) Width K Ablation: Success Rate')
axs[0, 0].grid(True, axis='y')

axs[1, 0].bar(range(len(k_labels)), k_p_iae, color='coral', alpha=0.8)
axs[1, 0].set_xticks(range(len(k_labels)))
axs[1, 0].set_xticklabels(k_labels, rotation=45)
axs[1, 0].set_ylabel('Position IAE (m·s)')
axs[1, 0].set_title('(a) Width K Ablation: Tracking Error')
axs[1, 0].grid(True, axis='y')

# σ 消融
s_data = data['sigma']
s_labels = list(s_data.keys())
s_success = [s_data[l]['success_rate'] for l in s_labels]
s_p_iae = [s_data[l]['p_iae_mean'] for l in s_labels]
axs[0, 1].plot(range(len(s_labels)), s_success, 'b-o', linewidth=2, markersize=8)
axs[0, 1].set_xticks(range(len(s_labels)))
axs[0, 1].set_xticklabels(s_labels, rotation=45)
axs[0, 1].set_ylabel('Success Rate (%)')
axs[0, 1].set_title('(b) Noise Scale σ Ablation: Success Rate')
axs[0, 1].grid(True)

axs[1, 1].plot(range(len(s_labels)), s_p_iae, 'r-s', linewidth=2, markersize=8)
axs[1, 1].set_xticks(range(len(s_labels)))
axs[1, 1].set_xticklabels(s_labels, rotation=45)
axs[1, 1].set_ylabel('Position IAE (m·s)')
axs[1, 1].set_title('(b) Noise Scale σ Ablation: Tracking Error')
axs[1, 1].grid(True)

# D 消融
d_data = data['D']
d_labels = list(d_data.keys())
d_success = [d_data[l]['success_rate'] for l in d_labels]
d_p_iae = [d_data[l]['p_iae_mean'] for l in d_labels]
axs[0, 2].plot(range(len(d_labels)), d_success, 'g-^', linewidth=2, markersize=8)
axs[0, 2].set_xticks(range(len(d_labels)))
axs[0, 2].set_xticklabels(d_labels, rotation=45)
axs[0, 2].set_ylabel('Success Rate (%)')
axs[0, 2].set_title('(c) Recursion Depth D Ablation: Success Rate')
axs[0, 2].grid(True)

axs[1, 2].plot(range(len(d_labels)), d_p_iae, 'm-d', linewidth=2, markersize=8)
axs[1, 2].set_xticks(range(len(d_labels)))
axs[1, 2].set_xticklabels(d_labels, rotation=45)
axs[1, 2].set_ylabel('Position IAE (m·s)')
axs[1, 2].set_title('(c) Recursion Depth D Ablation: Tracking Error')
axs[1, 2].grid(True)

plt.suptitle(f'Ablation Study Results ({num_trials} Monte Carlo Trials per Configuration)', fontsize=14, fontweight='bold')
plt.tight_layout()

fig_path = os.path.join(RESULTS_DIR, 'ptrm_nmpc_ablation.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
fig_path_pdf = os.path.join(RESULTS_DIR, 'ptrm_nmpc_ablation.pdf')
plt.savefig(fig_path_pdf, bbox_inches='tight')
print(f"\n消融实验图表已保存至: {fig_path} 和 {fig_path_pdf}")
plt.close(fig)

# CBF 和滞回消融对比图
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

cbf_data = data['CBF']
cbf_labels = list(cbf_data.keys())
cbf_success = [cbf_data[l]['success_rate'] for l in cbf_labels]
cbf_collision = [cbf_data[l]['collision_count'] for l in cbf_labels]
x_pos = np.arange(len(cbf_labels))
w = 0.35
axs2[0].bar(x_pos - w/2, cbf_success, w, label='Success Rate (%)', color='steelblue')
axs2[0].bar(x_pos + w/2, cbf_collision, w, label='Collision Count', color='coral')
axs2[0].set_xticks(x_pos)
axs2[0].set_xticklabels(cbf_labels)
axs2[0].set_title('(d) DT-CCBF Ablation')
axs2[0].legend()
axs2[0].grid(True, axis='y')

hyst_data = data['hysteresis']
hyst_labels = list(hyst_data.keys())
hyst_success = [hyst_data[l]['success_rate'] for l in hyst_labels]
hyst_p_iae = [hyst_data[l]['p_iae_mean'] for l in hyst_labels]
x_pos2 = np.arange(len(hyst_labels))
axs2[1].bar(x_pos2 - w/2, hyst_success, w, label='Success Rate (%)', color='steelblue')
axs2[1].bar(x_pos2 + w/2, hyst_p_iae, w, label='Position IAE', color='coral')
axs2[1].set_xticks(x_pos2)
axs2[1].set_xticklabels(hyst_labels)
axs2[1].set_title('(e) Hysteresis Penalty Ablation')
axs2[1].legend()
axs2[1].grid(True, axis='y')

plt.tight_layout()
fig2_path = os.path.join(RESULTS_DIR, 'ptrm_nmpc_ablation_cbf_hyst.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"CBF/滞回消融对比图已保存至: {fig2_path}")
