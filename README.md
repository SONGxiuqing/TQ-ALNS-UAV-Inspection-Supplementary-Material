# TQ-ALNS-UAV-Inspection

Source code and reproducibility package aligned with the revised manuscript on
heterogeneous multi-UAV inspection location-routing under energy-consumption uncertainty.

## Manuscript experiment map

- **Table 4** — benchmark scale settings (data/configuration only)
- **Table 5** — exact validation against Gurobi on small instances
- **Table 6** — time-limited Gurobi vs. TQ-ALNS scalability tests
- **Table 7** — homogeneous vs. heterogeneous UAV configuration
- **Table 8** — robustness over random instances
- **Table 9** — sensitivity-analysis parameter settings
- **Tables 10–12** — case-study data, UAV prototypes, and reported routes
- **Figures 3–5** — confidence-level, demand-scale, and battery-capacity sensitivity

The computational experiment section contains **Section 5.3 UAV configuration** and
**Section 5.4 robustness analysis**. There is no separate component-removal comparison
experiment in this release.

## Installation

```bash
pip install -r requirements.txt
```

A valid Gurobi license is required for the Gurobi experiments. Do not commit `gurobi.lic`.

## Official run commands

```bash
python experiments/run_table5_exact_validation.py
python experiments/run_table6_scalability.py
python experiments/run_table7_uav_configuration.py
python experiments/run_table8_robustness.py
python experiments/run_fig3_confidence_level.py
python experiments/run_fig4_demand_scale.py
python experiments/run_fig5_battery_capacity.py
python experiments/run_case_study.py
```

Run all three sensitivity groups together with:

```bash
python experiments/run_chapter6_all.py
```

## Repository structure

- `src/` — model and TQ-ALNS source modules
- `experiments/` — official manuscript-facing run entry points
- `configs/` — manuscript table snapshots and machine-readable experiment settings
- `data/case_study/` — standardized case-study data supplied during revision
- `results/` — generated outputs
- `tests/` — static/import checks

## Reproducibility notes

The displayed exact-validation table contains J=3–13. The package follows those displayed rows.
The case-study coordinate material supplied for public release contains 95 monitoring points,
five hazardous sources, and the five selected platform coordinates. The full set of 20 candidate
platform coordinates is required for complete 20-site re-optimization and is not fabricated here.
