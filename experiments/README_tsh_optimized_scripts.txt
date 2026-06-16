Optimized TSH figure scripts

Files:
- tsh_fig1_mechanism_optimized.py
- tsh_fig2_quant_optimized.py
- tsh_fig3_comparisons_optimized.py
- tsh_figS1_tasks_optimized.py
- tsh_figS2_tasks_3d_optional.py   (optional extra 3D figure)
- tsh_plot_style_optimized.py
- tsh_ptrm_advantage_quick.py      (dependency, original)

Notes:
1. The optimized scripts write to the same output paths as your current scripts:
   experiments/results_v6/*.pdf
2. The S1 and S2 task figures keep the same task-generation dependency based on
   tsh_ptrm_advantage_quick.py.
3. Recommended replacements in the manuscript:
   - Fig. 1: use tsh_fig1_mechanism_optimized.py
   - Fig. 2: use tsh_fig2_quant_optimized.py
   - Fig. 3: use tsh_fig3_comparisons_optimized.py
   - Fig. S1: use tsh_figS1_tasks_optimized.py
4. Optional add-on:
   - Fig. S2 3D benchmark overview: use tsh_figS2_tasks_3d_optional.py
