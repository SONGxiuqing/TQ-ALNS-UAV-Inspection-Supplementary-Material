# Manuscript-to-code map

| Manuscript item | Official command | Source |
|---|---|---|
| Table 4 benchmark settings | configuration only | `configs/Table4_benchmark_scale_settings.csv` |
| Table 5 exact validation | `python experiments/run_table5_exact_validation.py` | `src/table5_exact_validation.py` |
| Table 6 scalability | `python experiments/run_table6_scalability.py` | `src/table6_scalability_main.py` + `src/table6_scalability_supplement.py` |
| Table 7 UAV configuration | `python experiments/run_table7_uav_configuration.py` | `src/table7_uav_configuration.py` |
| Table 8 robustness | `python experiments/run_table8_robustness.py` | `src/table5_exact_validation.py` + `src/tqalns_benchmark_core.py` |
| Figure 3 confidence level | `python experiments/run_fig3_confidence_level.py` | `src/chapter6_sensitivity.py` |
| Figure 4 demand scale | `python experiments/run_fig4_demand_scale.py` | `src/chapter6_sensitivity.py` |
| Figure 5 battery capacity | `python experiments/run_fig5_battery_capacity.py` | `src/chapter6_sensitivity.py` |
| Section 7 case validation | `python experiments/run_case_study.py` | `src/case_study_validation.py` |
