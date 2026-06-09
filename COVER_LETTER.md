# Cover Letter

**Manuscript Title.** *Wide-Scale Sampling with Hard-Argmin Selection and CBF Projection for Safe Quadrotor Navigation in Narrow Passages*

**Authors.**
- Zhuang Shao¹,* — shaozhuang@crpower.com.cn (Corresponding author, ORCID: 0000-0003-2496-0797)
- Lijun Lei² — leilijun6@crpower.com.cn
- Peng Wang³ — wangpeng@ncwu.edu.cn
- Liang Zheng² — zhengliang35@crpower.com.cn
- Jie Zhou² — zhoujie365@crpower.com.cn

**Affiliations.**
1. China Resources Power Technology Research Institute Co., Ltd., Shenzhen 518000, Guangdong Province, China
2. Rundian Energy Science and Technology Co., Ltd., Zhengzhou 450052, Henan Province, China
3. North China University of Water Resources and Electric Power, Zhengzhou 450046, Henan Province, China

**Article Type.** Regular Paper

**Date.** 2026-06-08

---

Dear Editor-in-Chief,

We respectfully submit the manuscript *"Wide-Scale Sampling with Hard-Argmin Selection and CBF Projection for Safe Quadrotor Navigation in Narrow Passages"* for consideration as a Regular Paper.

## Significance of the Work

Sampling-based nonlinear MPC — MPPI, CEM, and their many variants — is widely deployed for quadrotor obstacle avoidance, yet it shares a structural weakness: the single-source Gaussian around the nominal control collapses when the nominal points into an obstacle, and adding samples around the same bad nominal does not help. This brittleness becomes catastrophic on narrow-passage tasks, where the single-source PD-Gaussian baseline succeeds on only $66.3\%$ of trials at $K = 10$ ($N_{MC} = 80$, paired McNemar).

We propose **TSH-NMPC** (Test-time Scaling with Hard-argmin), an intentionally simple alternative: $K$ wide-scale Gaussian samples ($\sigma = 5\,\mathrm{N}$) are drawn around the PD nominal, scored by a unified short closed-loop rollout, hard-argmin selected, and projected onto a DT-CCBF safety set. A two-source variant (mixing tight $\sigma=2$ and wide $\sigma=5$) is statistically equivalent to the single-source-wide default (paired McNemar $p = 0.51$), confirming the lever is the noise scale, not the source partition.

## Key Contributions

1. **A wide-scale sampling pool that strictly Pareto-dominates the single-source PD-Gaussian baseline on the studied narrow-passage benchmark.** Success rises from $66.3\%$ to $98.8\%$ at $K = 10$ on the narrow benchmark with $N_{MC} = 80$ paired trials, with McNemar exact $p < 0.001$ and bootstrap $95\%$ CIs on $\Delta\text{TErr}$ and $\Delta\text{IAE}$ excluding zero. The single-source-wide variant (all $K$ at $\sigma=5$) is statistically equivalent to the two-source mixture (McNemar $p=0.51$), confirming the lever is the noise scale. We compare against MPPI, CEM, and iCEM at matched $K$ and matched DT-CCBF; hard-argmin yields $3$--$4.4\times$ better $\TErr$ than soft-weighting or iterative refitting.

2. **Passive parameter adaptation without online identification.** Under a $+50\%$ mass mismatch, single-source PD collapses to $0/40$ success while TSH-NMPC retains $31/40 = 77.5\%$ — under matched paired seeds, *every one of the 40 trials where PD failed, TSH-NMPC succeeded* ($b = 31$, $c = 0$, $p < 0.001$). This is the strongest single piece of evidence in the paper.

3. **Hard-argmin selector with approximately deterministic compute budget.** TSH-NMPC at $K = 10$ runs at mean $10.7\,$ms per step (per-step profiling; p99 = $16.3\,$ms, 0.17% over 20ms deadline) or ${\approx}6\,$ms per-step median when amortized over a full trial. CasADi+IPOPT has lower mean latency ($5.2\,$ms) but data-dependent tails (p99 = $11.9\,$ms, max = $35.0\,$ms, 0.12% over deadline). TSH-NMPC's $\mathcal{O}(K \cdot S)$ compute is algorithmically data-independent (CV $< 20\%$), making it suitable for hard real-time platforms where deterministic execution is preferred.

4. **Controlled negative ablation of learned priors.** Replacing the random wide-scale samples with a $27{,}935$-parameter recursive neural predictor trained on $50{,}000$ closed-loop expert trajectories produces no statistically detectable advantage on this task class ($p \ge 0.18$ for every $K$ tested at $N_{MC} = 80$ paired trials). At $N_{MC} = 300$ the random wide source is statistically *superior* (McNemar $p = 7.4 \times 10^{-6}$). Retraining the predictor on task-matched data ($9\times$ open-loop improvement) degrades closed-loop performance further, indicating that the mechanism is driven by *diversity*, not by learned-prior accuracy. To our knowledge this is the first paper to report a controlled negative ablation of learned priors in sampling NMPC on this task class.

## Reproducibility and Code Availability

All numerical results in Tables II–VII come from JSON files produced by the open-source experiment pipeline. The complete code, raw JSON results, and figure-generation scripts are available at:

> https://github.com/VickylastShao/CBF-Projected-Hard-Argmin-Sampling-for-Safe-Quadrotor-Navigation

No table cell is hand-edited; bootstrap CIs use $10^4$ resamples; significance is reported with the McNemar exact two-sided test on paired discordant counts. The line-numbered manuscript PDF is provided to facilitate reviewer reference.

## Negative-Result Documentation

The learned-prior negative ablation (Section V-F) is reported with full statistical machinery rather than buried in a footnote. This kind of "what does not work" evidence is uncommon in the sampling-MPC literature and we believe it provides actionable guidance for practitioners considering learned warm-starts.

## Originality and Disclosure

This manuscript has not been published elsewhere and is not under consideration by any other journal. The author-constructed `narrow`/`two_gate`/`u_shape` benchmark is described in Section V-A and the raw JSON results plus the experiment scripts will be released upon acceptance via the repository above. We acknowledge the absence of a widely accepted third-party narrow-passage quadrotor benchmark as a limitation in Section VI-D.

## Suggested Reviewers

We respectfully suggest the following reviewers, each with relevant expertise in nonlinear control, MPC, and/or autonomous systems:

| Reviewer | Affiliation | E-mail |
|---|---|---|
| **Jianxin Zhou** | Southeast University | zjx@seu.edu.cn |
| **Zhenlong Wu** | Zhengzhou University | wuzhenlong2020@zzu.edu.cn |
| **Cong Yu** | Jianghan University | congy@jhun.edu.cn |
| **Hui Gu** | (Independent reviewer) | guhuini@126.com |

None of the suggested reviewers has co-authored with the authors in the past three years, nor shares an institutional affiliation with any author at the time of submission.

## Conflicts of Interest

The authors declare no conflicts of interest. No funding agency was involved in the design, execution, or interpretation of the work.

## Author Contributions

- **Zhuang Shao (Corresponding):** Conceptualization; methodology; software; formal analysis; writing — original draft; supervision.
- **Lijun Lei:** Methodology; experimental validation; writing — review & editing.
- **Peng Wang:** Theoretical analysis (DT-CCBF, monotone-min-cost); writing — review & editing.
- **Liang Zheng:** Software; data curation; visualization.
- **Jie Zhou:** Investigation; experimental setup; writing — review & editing.

We thank the editors and reviewers for their time and consideration.

Sincerely,

**Zhuang Shao**
Corresponding Author
China Resources Power Technology Research Institute Co., Ltd.
Shenzhen 518000, Guangdong Province, China
E-mail: shaozhuang@crpower.com.cn
ORCID: 0000-0003-2496-0797
