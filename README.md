# 6CCE3EGS Case B Grid Scale Battery Arbitrage and Carbon Aware Dispatch

This repository contains the Python implementation, generated outputs, and report for **Case B** of the 6CCE3EGS individual coursework. The project models a **grid scale battery** participating in the wholesale electricity market under two scopes:

- **Base case**: energy arbitrage using day-ahead prices
- **Extension**: carbon-aware dispatch using the same battery model with an added carbon penalty term

The work follows the coursework focus on **model formulation, explainable implementation, explicit verification, and engineering insight**.

---

## Project overview

The model represents a **2 MWh battery** with **1 MW charge and discharge limits**, **hourly time steps**, and **60 days of market data**. A simple rule-based schedule is first used as a baseline. A **CVXPY linear programming model** is then used to find the optimal arbitrage dispatch under the same physical constraints. The extension adds a **carbon-aware objective** to examine the trade-off between profit and emissions.

### Battery parameters

- Energy capacity: **2000 kWh**
- Maximum charge power: **1000 kW**
- Maximum discharge power: **1000 kW**
- Charging efficiency: **0.938**
- Discharging efficiency: **0.938**
- Initial SOC: **1000 kWh**
- Time step: **1 hour**

---

## Methods included

### 1. Simple rule simulation
A fixed daily schedule used as a baseline:
- charge during night hours
- discharge during evening hours
- enforce the terminal state of charge target

### 2. Optimal profit-only dispatch
A linear programming model that minimizes net electricity cost, which is equivalent to maximizing arbitrage profit.

### 3. Optimal carbon-aware dispatch
The same linear programming structure as the profit-only model, but with an added carbon penalty term:

- carbon price: **£0.05 per kg CO2**
- net carbon definition: **charging emissions minus discharge displacement**

---

## Mathematical structure

### State of charge update
\[
E_{t+1} = E_t + \eta_{ch} p_{ch,t} \Delta t - \frac{1}{\eta_{dis}} p_{dis,t} \Delta t
\]

### Operating limits
\[
0 \le E_t \le E_{max}
\]
\[
0 \le p_{ch,t} \le P_{ch,max}
\]
\[
0 \le p_{dis,t} \le P_{dis,max}
\]

### Profit-only objective
Minimize net electricity cost over the horizon:
\[
\min \sum_t \left( \pi_t^{DA} p_{ch,t} - \pi_t^{DA} p_{dis,t} \right) \Delta t
\]

### Carbon-aware objective
\[
\min \sum_t \left( \pi_t^{DA} p_{ch,t} - \pi_t^{DA} p_{dis,t} \right) \Delta t + c_{CO2} \sum_t \left( g_t p_{ch,t} - g_t p_{dis,t} \right) \Delta t
\]

where:
- \(\pi_t^{DA}\) is the day-ahead electricity price
- \(g_t\) is carbon intensity
- \(c_{CO2}\) is the virtual carbon price

---

## Repository contents

```text
.
├── caseB_grid_battery_coursework_main_and_extension.py
├── caseB_grid_battery_market_hourly.csv
├── caseB_run_summary.json
├── caseB_summary_metrics.csv
├── caseB_verification_checks.csv
├── caseB_hourly_dispatch_simple_rule.csv
├── caseB_hourly_dispatch_optimal_profit_only.csv
├── caseB_hourly_dispatch_optimal_carbon_aware.csv
├── caseB_mainline_cumulative_profit.png
├── caseB_mainline_price_and_dispatch_zoom.png
├── caseB_mainline_soc_comparison.png
├── caseB_extension_profit_vs_carbon_zoom.png
├── caseB_extension_tradeoff.png
├── k22036880 6CCE3EGS Individual Coursework Report.pdf
├── CW.pdf
└── README.md
```

---

## How to run

### Requirements

Install the required Python packages locally:

```bash
pip install pandas numpy matplotlib cvxpy
```

### Run the script

From the repository folder, run:

```bash
python caseB_grid_battery_coursework_main_and_extension.py
```

### Optional arguments

```bash
python caseB_grid_battery_coursework_main_and_extension.py \
  --input caseB_grid_battery_market_hourly.csv \
  --outdir caseB_outputs_main_and_extension \
  --carbon-price 0.05
```

---

## Input data

The model uses:

- `caseB_grid_battery_market_hourly.csv`

Expected columns:
- `timestamp`
- `day_ahead_price_gbp_per_mwh`
- `imbalance_price_gbp_per_mwh`
- `ancillary_availability_gbp_per_mw_per_h`
- `carbon_intensity_kg_per_kwh_optional`

### Unit conversion
Day-ahead prices are converted from **GBP/MWh** to **GBP/kWh**.

Example:

- **13.65 GBP/MWh = 0.01365 GBP/kWh**

---

## Outputs generated

### CSV outputs

- `caseB_hourly_dispatch_simple_rule.csv`
- `caseB_hourly_dispatch_optimal_profit_only.csv`
- `caseB_hourly_dispatch_optimal_carbon_aware.csv`
- `caseB_summary_metrics.csv`
- `caseB_verification_checks.csv`
- `caseB_run_summary.json`

### Figures

- `caseB_mainline_cumulative_profit.png`
- `caseB_mainline_price_and_dispatch_zoom.png`
- `caseB_mainline_soc_comparison.png`
- `caseB_extension_profit_vs_carbon_zoom.png`
- `caseB_extension_tradeoff.png`

---

## Headline results

### Base case comparison

| Method | Profit (£) | Charged (kWh) | Discharged (kWh) | Equivalent full cycles |
|---|---:|---:|---:|---:|
| Simple rule simulation | 3666.80 | 127931.77 | 112560.00 | 56.28 |
| Optimal profit-only | 16172.25 | 331671.31 | 291819.01 | 145.91 |

### Extension comparison

| Method | Profit (£) | Net carbon (kg CO2) | Objective value (£ equivalent) |
|---|---:|---:|---:|
| Optimal profit-only | 16172.25 | -7377.93 | 16172.25 |
| Optimal carbon-aware | 16035.80 | -13363.66 | 16703.98 |

### Key interpretation

- The optimization model increases profit substantially relative to the fixed rule baseline.
- The battery remains within its technical limits throughout the horizon.
- The carbon-aware extension achieves a much larger carbon reduction with only a small drop in profit.
- Profit loss from carbon-aware dispatch is **£136.45**, or **0.84%**.
- Additional carbon reduction is **5985.73 kg CO2**, an improvement of **81.1%** over the profit-only optimum.

---

## Verification checks

The project includes explicit verification rather than relying only on the solver output. Checks include:

- unit conversion accuracy
- SOC dynamic residuals
- SOC bounds
- charge and discharge power limits
- non-negative traded energy
- profit reconstruction accuracy
- no simultaneous charging and discharging
- terminal SOC satisfaction

The exported file `caseB_verification_checks.csv` contains the numerical results for all verification items.

---

## Modelling scope and limitations

This coursework model is intentionally simplified to keep the formulation transparent and verifiable.

### Included
- day-ahead arbitrage
- hourly SOC dynamics
- efficiency losses
- end-of-horizon SOC target
- carbon-aware objective in the extension

### Not included
- battery degradation in the final model
- forecasting error
- bid-ask spread effects
- uncertainty handling
- ancillary market participation in the dispatch decision
- imbalance market participation in the dispatch decision

Note: the ancillary and imbalance columns are present in the dataset, but they are **not used** in the current model. The carbon column is used **only in the extension objective**.

---

## Report

The full coursework report is included in this repository:

- `k22036880 6CCE3EGS Individual Coursework Report.pdf`

The coursework brief is also included for reference:

- `CW.pdf`

---

## AI usage statement

As stated in the report, ChatGPT was used during report drafting to help organise the report structure. It was **not used for code or data processing** in the submitted assignment version.

---

## Repository reference used in the report

- Repository: `6CCE3EGS`
- URL: `https://github.com/wsl0227/6CCE3EGS`
- Version referenced in report: `93cd7e6`
- Date accessed in report: `6 April 2026`

---

## Author

**Shengli Wang**

6CCE3EGS Individual Coursework
