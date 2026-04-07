# 6CCE3EGS Case B Coursework

This repository contains my solution for **Case B: Grid Scale Battery in Electricity Markets**.

The project models a **2 MWh battery** with **1 MW charging and discharging limits** using hourly electricity market data. It compares a **simple rule-based dispatch** with an **optimisation-based dispatch** for energy arbitrage. It also includes a **carbon-aware extension** to study the trade-off between profit and emissions.

## Repository contents

- `caseB_grid_battery_coursework_main_and_extension.py`  
  Main Python script for the base case and extension

- `caseB_grid_battery_market_hourly.csv`  
  Input dataset

- `k22036880 6CCE3EGS Individual Coursework Report.pdf`  
  Final report

- `caseB_summary_metrics.csv`  
  Summary results for all methods

- `caseB_verification_checks.csv`  
  Verification results

- `caseB_hourly_dispatch_simple_rule.csv`  
  Hourly dispatch output for the simple rule

- `caseB_hourly_dispatch_optimal_profit_only.csv`  
  Hourly dispatch output for the profit-only optimisation

- `caseB_hourly_dispatch_optimal_carbon_aware.csv`  
  Hourly dispatch output for the carbon-aware extension

## Model setup

The battery parameters used in this project are:

- Energy capacity: **2000 kWh**
- Maximum charge power: **1000 kW**
- Maximum discharge power: **1000 kW**
- Charging efficiency: **0.938**
- Discharging efficiency: **0.938**
- Initial state of charge: **1000 kWh**
- Time step: **1 hour**

## Methods

### Base case
Two dispatch methods are compared:

1. **Simple rule-based simulation**  
   A fixed charging and discharging schedule

2. **Optimal profit-only dispatch**  
   A linear programming model that maximises arbitrage profit using day-ahead prices

### Extension
The extension adds a **carbon-aware objective** to the same battery model.  
It applies a carbon penalty term so that the optimiser considers both **financial profit** and **carbon performance**.

## How to run

Install the required packages:

```bash
pip install pandas numpy matplotlib cvxpy
