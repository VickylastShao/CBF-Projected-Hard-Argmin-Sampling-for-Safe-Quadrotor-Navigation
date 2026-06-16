# Cover Letter — IJRA Submission

**Manuscript Title.** *Wide-Scale Sampling with Hard-Argmin Selection and CBF Projection for Safe Quadrotor Navigation in Narrow Passages*

**Authors.**
- Zhuang Shao¹,* — shaozhuang@crpower.com.cn (Corresponding author, ORCID: 0000-0003-2496-0797)
- Lijun Lei¹ — leilijun@crpower.com.cn
- Peng Wang¹ — wangpeng@crpower.com.cn
- Liang Zheng¹ — zhengliang@crpower.com.cn
- Jie Zhou² — zhoujie@crpower.com.cn

**Affiliations.**
1. China Resources Power Technology Research Institute Co., Ltd., Shenzhen 518000, China
2. China Resources Power Holdings Co., Ltd., Shenzhen 518000, China

**Article Type.** Regular Paper

**Date.** June 2026

---

Dear Editor-in-Chief of the International Journal of Robotics & Automation,

We respectfully submit the manuscript *"Wide-Scale Sampling with Hard-Argmin Selection and CBF Projection for Safe Quadrotor Navigation in Narrow Passages"* for consideration as a Regular Paper in the International Journal of Robotics & Automation.

## Match to IJRA Scope

This paper tackles a practical problem in robot navigation: getting sampling-based nonlinear model predictive control (NMPC) to work reliably in tight passages, where the usual baseline policy is structurally biased. The controller we put forward --- TSH-NMPC --- puts together wide-scale Gaussian sampling, hard-argmin selection, and a discrete-time control barrier function (DT-CCBF) safety projection. It reaches success-rate parity with an ideal NLP-based NMPC while running at roughly 6 ms per step on a single CPU core, with no GPU or NLP-solver dependency. The three pieces each play an empirically necessary role and work jointly within the narrow-passage quadrotor setting studied here.

We see the work fitting IJRA's scope in three ways:
- **Robot control and planning:** the paper describes a practical controller architecture for real-time obstacle avoidance on resource-constrained platforms, checked against a Monte Carlo benchmark (up to N=300 paired trials) across three task families.
- **Safety and verification:** the DT-CCBF projection supplies hard collision-avoidance guarantees, and the paper includes a dedicated CBF fallback audit (N=40) that found zero collisions even under emergency fallback.
- **Reproducible methodology:** every numerical result comes from an open-source pipeline; the full code, raw JSON results, and figure-generation scripts sit in a public repository.

## Significance

Sampling-based NMPC (MPPI, CEM and variants) is widely used for quadrotor obstacle avoidance, but it shares a structural weak point: when the nominal control points into an obstacle, every candidate picks up the same bias, and adding more samples does not fix it. On our narrow-passage benchmark, the usual single-source PD-Gaussian baseline succeeds on only 66.3% of trials at K=10 (N=80 paired McNemar). TSH-NMPC lifts this to 98.8% with a deliberately simple change: wider Gaussian noise (σ=5 per axis) at the same K, hard-argmin selection (no soft weighting), and CBF post-projection.

## Key Results

1. **Main narrow-passage benchmark (N=80):** TSH-NMPC at K=10 reaches ≥93% success against 63–70% for single-source PD (McNemar p < 0.001). Taking out any of the three pieces makes the controller fall apart.

2. **Passive robustness to model mismatch:** Under +50% mass mismatch (N=40), PD falls to 0% while TSH-NMPC keeps ≥77.5% success. In every one of the 40 trials where PD failed, TSH-NMPC succeeded (b=31, c=0, p < 0.001).

3. **NLP-free success-rate parity:** Against CasADi+IPOPT collocation NMPC at horizon H=20, TSH-NMPC reaches success-rate parity (McNemar p=1.0) at roughly 6 ms per step with algorithmically deterministic compute (coefficient of variation < 20%).

4. **Diversity-over-accuracy ablation:** A controlled negative ablation at N=300 finds a 27,935-parameter learned predictor clearly worse than a random wide source (p = 7.4 × 10⁻⁶), pointing to sampling diversity as the main mechanism.

## Reproducibility and Code Availability

All the numbers in the manuscript come from raw JSON files produced by the open-source experiment pipeline. The full code, raw JSON data, and figure-generation scripts are available at:

> https://github.com/VickylastShao/CBF-Projected-Hard-Argmin-Sampling-for-Safe-Quadrotor-Navigation

Every table cell is rebuilt from JSON by the audit script. Bootstrap CIs use 10⁴ resamples; significance is reported with the McNemar exact two-sided test on paired discordant counts.

## Ethics and Originality

This manuscript has not been published elsewhere and is not under review at any other journal. The authors declare no conflict of interest. No funding agency took part in the design, running, or interpretation of the work.

## Suggested Reviewers

The following two researchers have relevant expertise in non-linear control, model predictive control, and autonomous systems, and have not co-authored with any member of our group in the past three years:

1. **Prof. Jianxin Zhou** — School of Automation, Southeast University, Nanjing, China
   - Email: zjx@seu.edu.cn
   - Expertise: Non-linear control, predictive control, unmanned aerial vehicle systems

2. **Prof. Zhenlong Wu** — School of Electrical and Information Engineering, Zhengzhou University, Zhengzhou, China
   - Email: wuzhenlong2020@zzu.edu.cn
   - Expertise: Robust control, model predictive control, industrial automation

## Author Contributions

- **Zhuang Shao (Corresponding author):** Conceptualisation; methodology; software; formal analysis; writing — original draft; supervision.
- **Lijun Lei:** Methodology; experimental validation; writing — review & editing.
- **Peng Wang:** Theoretical analysis (DT-CCBF, monotone min-cost); writing — review & editing.
- **Liang Zheng:** Software; data curation; visualisation.
- **Jie Zhou:** Investigation; experimental setup; writing — review & editing.

We thank the editors and reviewers for their time and consideration.

Sincerely,

**Zhuang Shao**
Corresponding Author
China Resources Power Technology Research Institute Co., Ltd.
Shenzhen 518000, China
E-mail: shaozhuang@crpower.com.cn
ORCID: 0000-0003-2496-0797