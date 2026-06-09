# PTRM Advantage Quick Benchmark Summary
## 主要结论
- 纯 TRM+Rollout 在当前已训练模型和三个 procedural tasks 上均失败，不能作为 PTRM-positive 证据。
- TRM-heavy 混合（alpha=0.3 或 0.8）会破坏 PD 稳定性；alpha=0.95 的轻量 TRM bias 是目前唯一有希望的配置。
- 聚焦复验显示 narrow 场景中 alpha=0.95 的 TRM+PD+Rollout 相比 PD+Rollout 有小但一致的优势：K=5 成功率 70% vs 65%，TErr 0.373 vs 0.444，IAE 2.934 vs 3.012；K=10 成功率 85% vs 75%，TErr 0.294 vs 0.424，IAE 2.662 vs 2.826，同时 latency 更低。
- D-sweep 支持“存在最佳递归深度”的弱信号，D=16/K=10 最强；但不支持单调 D-scaling。
- CEM 在若干设置中 IAE 更好，但使用 effective rollouts=3K 且 latency 约 19ms，高于 TRM+PD 的约 9-10ms。

## 关键表格
### ptrm_advantage_quick.json
#### two_gate
- D16/K1
  - PD+Rollout: success=100.0%, TErr=0.069, IAE=1.529, lat_med=8.81ms, candBest=4691.4
  - TRM+Rollout: success=0.0%, TErr=3.824, IAE=2.970, lat_med=1.71ms, candBest=4708.3
  - TRM+PD+Rollout: success=0.0%, TErr=2.950, IAE=2.198, lat_med=1.69ms, candBest=4700.9
  - CEM: success=100.0%, TErr=0.068, IAE=1.531, lat_med=18.72ms, candBest=NA
- D16/K5
  - PD+Rollout: success=100.0%, TErr=0.041, IAE=1.446, lat_med=14.65ms, candBest=4691.3
  - TRM+Rollout: success=0.0%, TErr=3.145, IAE=2.641, lat_med=9.23ms, candBest=4702.8
  - TRM+PD+Rollout: success=0.0%, TErr=3.013, IAE=2.152, lat_med=8.96ms, candBest=4699.7
  - CEM: success=100.0%, TErr=0.085, IAE=1.539, lat_med=19.97ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.433, lat_med=15.08ms, candBest=4691.3
  - TRM+Rollout: success=0.0%, TErr=2.988, IAE=2.507, lat_med=9.32ms, candBest=4703.0
  - TRM+PD+Rollout: success=0.0%, TErr=2.997, IAE=2.129, lat_med=9.65ms, candBest=4695.9
  - CEM: success=100.0%, TErr=0.088, IAE=1.516, lat_med=19.30ms, candBest=NA

#### narrow
- D16/K1
  - PD+Rollout: success=0.0%, TErr=4.503, IAE=4.550, lat_med=8.47ms, candBest=9246.5
  - TRM+Rollout: success=0.0%, TErr=7.252, IAE=4.588, lat_med=1.68ms, candBest=9059.8
  - TRM+PD+Rollout: success=0.0%, TErr=3.501, IAE=3.256, lat_med=1.88ms, candBest=9136.8
  - CEM: success=0.0%, TErr=4.493, IAE=4.548, lat_med=18.53ms, candBest=NA
- D16/K5
  - PD+Rollout: success=80.0%, TErr=0.329, IAE=3.029, lat_med=15.03ms, candBest=9231.2
  - TRM+Rollout: success=0.0%, TErr=5.435, IAE=3.970, lat_med=9.60ms, candBest=8992.4
  - TRM+PD+Rollout: success=0.0%, TErr=3.922, IAE=3.079, lat_med=10.01ms, candBest=9084.2
  - CEM: success=80.0%, TErr=0.337, IAE=2.888, lat_med=19.47ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.123, IAE=2.686, lat_med=15.33ms, candBest=9206.6
  - TRM+Rollout: success=0.0%, TErr=5.197, IAE=3.825, lat_med=8.99ms, candBest=8970.3
  - TRM+PD+Rollout: success=0.0%, TErr=3.889, IAE=3.042, lat_med=9.57ms, candBest=9037.3
  - CEM: success=100.0%, TErr=0.157, IAE=2.229, lat_med=19.28ms, candBest=NA

#### u_shape
- D16/K1
  - PD+Rollout: success=100.0%, TErr=0.048, IAE=1.215, lat_med=10.47ms, candBest=2389.9
  - TRM+Rollout: success=0.0%, TErr=3.946, IAE=2.801, lat_med=1.71ms, candBest=2367.7
  - TRM+PD+Rollout: success=0.0%, TErr=3.012, IAE=2.142, lat_med=1.70ms, candBest=2373.5
  - CEM: success=100.0%, TErr=0.047, IAE=1.213, lat_med=25.12ms, candBest=NA
- D16/K5
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.138, lat_med=16.61ms, candBest=2378.2
  - TRM+Rollout: success=0.0%, TErr=3.385, IAE=2.576, lat_med=11.28ms, candBest=2361.5
  - TRM+PD+Rollout: success=0.0%, TErr=3.064, IAE=2.069, lat_med=11.40ms, candBest=2367.8
  - CEM: success=100.0%, TErr=0.065, IAE=1.139, lat_med=25.20ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.108, lat_med=17.98ms, candBest=2376.6
  - TRM+Rollout: success=0.0%, TErr=3.163, IAE=2.469, lat_med=11.20ms, candBest=2357.0
  - TRM+PD+Rollout: success=0.0%, TErr=3.083, IAE=2.052, lat_med=11.93ms, candBest=2363.0
  - CEM: success=100.0%, TErr=0.063, IAE=1.103, lat_med=24.44ms, candBest=NA

### ptrm_advantage_alpha08.json
#### two_gate
- D16/K1
  - PD+Rollout: success=100.0%, TErr=0.069, IAE=1.529, lat_med=8.17ms, candBest=4691.4
  - TRM+Rollout: success=0.0%, TErr=3.824, IAE=2.970, lat_med=1.73ms, candBest=4708.3
  - TRM+PD+Rollout: success=0.0%, TErr=0.982, IAE=1.704, lat_med=1.82ms, candBest=4691.6
  - CEM: success=100.0%, TErr=0.068, IAE=1.531, lat_med=18.41ms, candBest=NA
- D16/K5
  - PD+Rollout: success=100.0%, TErr=0.041, IAE=1.446, lat_med=14.11ms, candBest=4691.3
  - TRM+Rollout: success=0.0%, TErr=3.145, IAE=2.641, lat_med=9.36ms, candBest=4702.8
  - TRM+PD+Rollout: success=0.0%, TErr=0.541, IAE=1.518, lat_med=9.27ms, candBest=4691.6
  - CEM: success=100.0%, TErr=0.085, IAE=1.539, lat_med=19.06ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.433, lat_med=14.85ms, candBest=4691.3
  - TRM+Rollout: success=0.0%, TErr=2.988, IAE=2.507, lat_med=9.63ms, candBest=4703.0
  - TRM+PD+Rollout: success=100.0%, TErr=0.421, IAE=1.453, lat_med=8.97ms, candBest=4691.4
  - CEM: success=100.0%, TErr=0.088, IAE=1.516, lat_med=19.31ms, candBest=NA

#### narrow
- D16/K1
  - PD+Rollout: success=0.0%, TErr=4.503, IAE=4.550, lat_med=8.82ms, candBest=9246.5
  - TRM+Rollout: success=0.0%, TErr=7.252, IAE=4.588, lat_med=1.74ms, candBest=9059.8
  - TRM+PD+Rollout: success=0.0%, TErr=2.147, IAE=3.971, lat_med=2.03ms, candBest=9245.9
  - CEM: success=0.0%, TErr=4.493, IAE=4.548, lat_med=18.44ms, candBest=NA
- D16/K5
  - PD+Rollout: success=80.0%, TErr=0.329, IAE=3.029, lat_med=15.64ms, candBest=9231.2
  - TRM+Rollout: success=0.0%, TErr=5.435, IAE=3.970, lat_med=9.00ms, candBest=8992.4
  - TRM+PD+Rollout: success=0.0%, TErr=0.649, IAE=2.497, lat_med=9.29ms, candBest=9201.3
  - CEM: success=80.0%, TErr=0.337, IAE=2.888, lat_med=19.67ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.123, IAE=2.686, lat_med=15.62ms, candBest=9206.6
  - TRM+Rollout: success=0.0%, TErr=5.197, IAE=3.825, lat_med=10.00ms, candBest=8970.3
  - TRM+PD+Rollout: success=20.0%, TErr=0.521, IAE=2.364, lat_med=9.67ms, candBest=9166.5
  - CEM: success=100.0%, TErr=0.157, IAE=2.229, lat_med=18.80ms, candBest=NA

#### u_shape
- D16/K1
  - PD+Rollout: success=100.0%, TErr=0.048, IAE=1.215, lat_med=11.07ms, candBest=2389.9
  - TRM+Rollout: success=0.0%, TErr=3.946, IAE=2.801, lat_med=1.86ms, candBest=2367.7
  - TRM+PD+Rollout: success=0.0%, TErr=0.947, IAE=1.425, lat_med=2.00ms, candBest=2384.9
  - CEM: success=100.0%, TErr=0.047, IAE=1.213, lat_med=24.39ms, candBest=NA
- D16/K5
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.138, lat_med=18.44ms, candBest=2378.2
  - TRM+Rollout: success=0.0%, TErr=3.385, IAE=2.576, lat_med=11.27ms, candBest=2361.5
  - TRM+PD+Rollout: success=40.0%, TErr=0.517, IAE=1.235, lat_med=11.17ms, candBest=2377.4
  - CEM: success=100.0%, TErr=0.065, IAE=1.139, lat_med=25.83ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.108, lat_med=17.52ms, candBest=2376.6
  - TRM+Rollout: success=0.0%, TErr=3.163, IAE=2.469, lat_med=11.01ms, candBest=2357.0
  - TRM+PD+Rollout: success=100.0%, TErr=0.402, IAE=1.195, lat_med=11.03ms, candBest=2372.2
  - CEM: success=100.0%, TErr=0.063, IAE=1.103, lat_med=25.91ms, candBest=NA

### ptrm_advantage_alpha095.json
#### two_gate
- D16/K1
  - PD+Rollout: success=100.0%, TErr=0.069, IAE=1.529, lat_med=8.54ms, candBest=4691.4
  - TRM+Rollout: success=0.0%, TErr=3.824, IAE=2.970, lat_med=1.82ms, candBest=4708.3
  - TRM+PD+Rollout: success=100.0%, TErr=0.253, IAE=1.529, lat_med=2.15ms, candBest=4691.4
  - CEM: success=100.0%, TErr=0.068, IAE=1.531, lat_med=19.00ms, candBest=NA
- D16/K5
  - PD+Rollout: success=100.0%, TErr=0.041, IAE=1.446, lat_med=14.40ms, candBest=4691.3
  - TRM+Rollout: success=0.0%, TErr=3.145, IAE=2.641, lat_med=9.38ms, candBest=4702.8
  - TRM+PD+Rollout: success=100.0%, TErr=0.132, IAE=1.440, lat_med=9.32ms, candBest=4691.3
  - CEM: success=100.0%, TErr=0.085, IAE=1.539, lat_med=19.19ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.433, lat_med=15.67ms, candBest=4691.3
  - TRM+Rollout: success=0.0%, TErr=2.988, IAE=2.507, lat_med=8.83ms, candBest=4703.0
  - TRM+PD+Rollout: success=100.0%, TErr=0.083, IAE=1.399, lat_med=9.21ms, candBest=4691.3
  - CEM: success=100.0%, TErr=0.088, IAE=1.516, lat_med=19.42ms, candBest=NA

#### narrow
- D16/K1
  - PD+Rollout: success=0.0%, TErr=4.503, IAE=4.550, lat_med=8.93ms, candBest=9246.5
  - TRM+Rollout: success=0.0%, TErr=7.252, IAE=4.588, lat_med=1.80ms, candBest=9059.8
  - TRM+PD+Rollout: success=0.0%, TErr=4.433, IAE=4.542, lat_med=1.96ms, candBest=9246.5
  - CEM: success=0.0%, TErr=4.493, IAE=4.548, lat_med=17.76ms, candBest=NA
- D16/K5
  - PD+Rollout: success=80.0%, TErr=0.329, IAE=3.029, lat_med=15.83ms, candBest=9231.2
  - TRM+Rollout: success=0.0%, TErr=5.435, IAE=3.970, lat_med=8.96ms, candBest=8992.4
  - TRM+PD+Rollout: success=100.0%, TErr=0.179, IAE=2.672, lat_med=9.01ms, candBest=9226.2
  - CEM: success=80.0%, TErr=0.337, IAE=2.888, lat_med=18.92ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.123, IAE=2.686, lat_med=15.12ms, candBest=9206.6
  - TRM+Rollout: success=0.0%, TErr=5.197, IAE=3.825, lat_med=8.95ms, candBest=8970.3
  - TRM+PD+Rollout: success=100.0%, TErr=0.135, IAE=2.653, lat_med=9.46ms, candBest=9201.7
  - CEM: success=100.0%, TErr=0.157, IAE=2.229, lat_med=18.08ms, candBest=NA

#### u_shape
- D16/K1
  - PD+Rollout: success=100.0%, TErr=0.048, IAE=1.215, lat_med=10.58ms, candBest=2389.9
  - TRM+Rollout: success=0.0%, TErr=3.946, IAE=2.801, lat_med=1.72ms, candBest=2367.7
  - TRM+PD+Rollout: success=100.0%, TErr=0.229, IAE=1.221, lat_med=2.13ms, candBest=2388.6
  - CEM: success=100.0%, TErr=0.047, IAE=1.213, lat_med=23.69ms, candBest=NA
- D16/K5
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.138, lat_med=16.85ms, candBest=2378.2
  - TRM+Rollout: success=0.0%, TErr=3.385, IAE=2.576, lat_med=11.04ms, candBest=2361.5
  - TRM+PD+Rollout: success=100.0%, TErr=0.120, IAE=1.111, lat_med=10.46ms, candBest=2380.8
  - CEM: success=100.0%, TErr=0.065, IAE=1.139, lat_med=25.15ms, candBest=NA
- D16/K10
  - PD+Rollout: success=100.0%, TErr=0.027, IAE=1.108, lat_med=17.61ms, candBest=2376.6
  - TRM+Rollout: success=0.0%, TErr=3.163, IAE=2.469, lat_med=11.71ms, candBest=2357.0
  - TRM+PD+Rollout: success=100.0%, TErr=0.076, IAE=1.106, lat_med=11.16ms, candBest=2375.5
  - CEM: success=100.0%, TErr=0.063, IAE=1.103, lat_med=24.33ms, candBest=NA

### ptrm_advantage_narrow_alpha095_n20.json
#### narrow
- D16/K5
  - PD+Rollout: success=65.0%, TErr=0.444, IAE=3.012, lat_med=15.48ms, candBest=8983.4
  - TRM+Rollout: success=0.0%, TErr=5.517, IAE=4.032, lat_med=8.95ms, candBest=8811.5
  - TRM+PD+Rollout: success=70.0%, TErr=0.373, IAE=2.934, lat_med=9.38ms, candBest=8976.5
  - CEM: success=75.0%, TErr=0.505, IAE=2.919, lat_med=19.05ms, candBest=NA
- D16/K10
  - PD+Rollout: success=75.0%, TErr=0.424, IAE=2.826, lat_med=15.35ms, candBest=8976.5
  - TRM+Rollout: success=0.0%, TErr=5.189, IAE=3.901, lat_med=8.82ms, candBest=8787.9
  - TRM+PD+Rollout: success=85.0%, TErr=0.294, IAE=2.662, lat_med=9.54ms, candBest=8973.7
  - CEM: success=80.0%, TErr=0.270, IAE=2.386, lat_med=19.43ms, candBest=NA

### ptrm_advantage_narrow_d_sweep.json
#### narrow
- D8/K5
  - PD+Rollout: success=80.0%, TErr=0.565, IAE=2.914, lat_med=13.38ms, candBest=8993.7
  - TRM+Rollout: success=0.0%, TErr=5.367, IAE=3.936, lat_med=8.18ms, candBest=8789.8
  - TRM+PD+Rollout: success=90.0%, TErr=0.308, IAE=2.728, lat_med=8.91ms, candBest=8993.6
  - CEM: success=70.0%, TErr=0.609, IAE=2.888, lat_med=19.14ms, candBest=NA
- D8/K10
  - PD+Rollout: success=100.0%, TErr=0.088, IAE=2.496, lat_med=13.58ms, candBest=8980.4
  - TRM+Rollout: success=0.0%, TErr=4.871, IAE=3.715, lat_med=8.18ms, candBest=8789.7
  - TRM+PD+Rollout: success=90.0%, TErr=0.206, IAE=2.497, lat_med=8.43ms, candBest=8985.9
  - CEM: success=100.0%, TErr=0.165, IAE=2.240, lat_med=19.66ms, candBest=NA
- D16/K5
  - PD+Rollout: success=80.0%, TErr=0.311, IAE=2.831, lat_med=14.68ms, candBest=8995.5
  - TRM+Rollout: success=0.0%, TErr=5.300, IAE=3.918, lat_med=9.06ms, candBest=8814.0
  - TRM+PD+Rollout: success=90.0%, TErr=0.261, IAE=2.793, lat_med=9.45ms, candBest=8984.8
  - CEM: success=80.0%, TErr=0.506, IAE=2.869, lat_med=19.54ms, candBest=NA
- D16/K10
  - PD+Rollout: success=70.0%, TErr=0.454, IAE=2.771, lat_med=15.35ms, candBest=8989.6
  - TRM+Rollout: success=0.0%, TErr=4.994, IAE=3.782, lat_med=9.28ms, candBest=8781.6
  - TRM+PD+Rollout: success=100.0%, TErr=0.139, IAE=2.476, lat_med=9.06ms, candBest=8985.2
  - CEM: success=90.0%, TErr=0.194, IAE=2.265, lat_med=19.65ms, candBest=NA
- D24/K5
  - PD+Rollout: success=100.0%, TErr=0.193, IAE=2.814, lat_med=16.78ms, candBest=8990.1
  - TRM+Rollout: success=0.0%, TErr=5.472, IAE=3.988, lat_med=9.74ms, candBest=8820.1
  - TRM+PD+Rollout: success=90.0%, TErr=0.392, IAE=2.794, lat_med=10.00ms, candBest=8996.5
  - CEM: success=70.0%, TErr=0.518, IAE=2.808, lat_med=19.27ms, candBest=NA
- D24/K10
  - PD+Rollout: success=90.0%, TErr=0.266, IAE=2.582, lat_med=17.37ms, candBest=8990.3
  - TRM+Rollout: success=0.0%, TErr=4.998, IAE=3.793, lat_med=9.43ms, candBest=8790.9
  - TRM+PD+Rollout: success=90.0%, TErr=0.318, IAE=2.561, lat_med=10.04ms, candBest=8977.5
  - CEM: success=100.0%, TErr=0.148, IAE=2.217, lat_med=18.89ms, candBest=NA

