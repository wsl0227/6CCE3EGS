from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError as exc:  
    cp = None
    _CVXPY_IMPORT_ERROR = exc
else:
    _CVXPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class BatteryParams:
    energy_capacity_kwh: float = 2000.0
    max_charge_power_kw: float = 1000.0
    max_discharge_power_kw: float = 1000.0
    eta_ch: float = 0.938
    eta_dis: float = 0.938
    initial_soc_kwh: float = 1000.0
    dt_hours: float = 1.0


@dataclass(frozen=True)
class DispatchMetrics:
    method: str
    total_profit_gbp: float
    gross_revenue_gbp: float
    gross_purchase_cost_gbp: float
    charged_energy_kwh: float
    discharged_energy_kwh: float
    total_throughput_kwh: float
    final_soc_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    equivalent_full_cycles: float
    terminal_soc_gap_kwh: float
    net_carbon_kg: float
    objective_value_gbp_equiv: float


REQUIRED_COLUMNS = {
    'timestamp',
    'day_ahead_price_gbp_per_mwh',
    'imbalance_price_gbp_per_mwh',
    'ancillary_availability_gbp_per_mw_per_h',
    'carbon_intensity_kg_per_kwh_optional',
}


def load_case_b(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    out = df.copy()
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=False)
    out = out.sort_values('timestamp').reset_index(drop=True)
    out['day_ahead_price_gbp_per_kwh'] = out['day_ahead_price_gbp_per_mwh'] / 1000.0
    out['carbon_intensity_kg_per_kwh'] = out['carbon_intensity_kg_per_kwh_optional'].astype(float)
    if out['carbon_intensity_kg_per_kwh'].isna().any():
        raise ValueError('carbon_intensity_kg_per_kwh_optional contains missing values.')
    out['hour'] = out['timestamp'].dt.hour
    return out


def _reachable_soc_corridor(remaining_steps_after_current: int, battery: BatteryParams, target_kwh: float) -> tuple[float, float]:
    max_future_charge = remaining_steps_after_current * battery.eta_ch * battery.max_charge_power_kw * battery.dt_hours
    max_future_discharge = remaining_steps_after_current * battery.max_discharge_power_kw * battery.dt_hours / battery.eta_dis
    low = max(0.0, target_kwh - max_future_charge)
    high = min(battery.energy_capacity_kwh, target_kwh + max_future_discharge)
    return low, high


def simple_rule_dispatch(
    df: pd.DataFrame,
    battery: BatteryParams,
    *,
    terminal_target_kwh: float | None = None,
    charge_hours: tuple[int, ...] = (1, 2, 3, 4, 5),
    discharge_hours: tuple[int, ...] = (17, 18, 19, 20),
)  ->  pd.DataFrame:
    if terminal_target_kwh is None:
        terminal_target_kwh = battery.initial_soc_kwh
    T = len(df)
    dt = battery.dt_hours
    soc = np.zeros(T + 1)
    soc[0] = battery.initial_soc_kwh
    charge = np.zeros(T)
    discharge = np.zeros(T)
    action = np.full(T, 'IDLE', dtype=object)
    reason = np.full(T, 'outside fixed schedule', dtype=object)

    for t in range(T):
        hour = int(df.loc[t, 'hour'])
        soc_now = float(soc[t])
        can_charge_kw = min(
            battery.max_charge_power_kw,
            max(0.0, (battery.energy_capacity_kwh - soc_now) / (battery.eta_ch * dt)),
        )
        can_discharge_kw = min(
            battery.max_discharge_power_kw,
            max(0.0, soc_now * battery.eta_dis / dt),
        )

        planned_charge = 0.0
        planned_discharge = 0.0
        if hour in charge_hours and can_charge_kw > 0.0:
            planned_charge = can_charge_kw
            action[t] = 'CHARGE'
            reason[t] = 'fixed night charging window'
        elif hour in discharge_hours and can_discharge_kw > 0.0:
            planned_discharge = can_discharge_kw
            action[t] = 'DISCHARGE'
            reason[t] = 'fixed evening discharge window'

        next_soc = soc_now + battery.eta_ch * planned_charge * dt - planned_discharge * dt / battery.eta_dis
        remaining_steps = T - (t + 1)
        low, high = _reachable_soc_corridor(remaining_steps, battery, terminal_target_kwh)
        next_soc = min(max(next_soc, low), high)
        delta_soc = next_soc - soc_now

        if delta_soc > 1e-12:
            charge[t] = min(battery.max_charge_power_kw, delta_soc / (battery.eta_ch * dt))
            discharge[t] = 0.0
            if action[t] != 'CHARGE':
                action[t] = 'CHARGE'
                reason[t] = 'terminal target correction via charging'
        elif delta_soc < -1e-12:
            charge[t] = 0.0
            discharge[t] = min(battery.max_discharge_power_kw, (-delta_soc) * battery.eta_dis / dt)
            if action[t] != 'DISCHARGE':
                action[t] = 'DISCHARGE'
                reason[t] = 'terminal target correction via discharging'
        else:
            charge[t] = 0.0
            discharge[t] = 0.0

        soc[t + 1] = soc_now + battery.eta_ch * charge[t] * dt - discharge[t] * dt / battery.eta_dis
        soc[t + 1] = min(max(soc[t + 1], 0.0), battery.energy_capacity_kwh)

    if abs(soc[-1] - terminal_target_kwh) > 1e-6:
        raise RuntimeError('Simple rule dispatch did not meet the terminal state-of-charge target.')

    return pd.DataFrame({
        'charge_kw': charge,
        'discharge_kw': discharge,
        'soc_start_kwh': soc[:-1],
        'soc_end_kwh': soc[1:],
        'action': action,
        'reason': reason,
    })


def _solve_cvxpy_problem(problem: 'cp.Problem') -> str:
    solver_attempts = [
        ('HiGHS', getattr(cp, 'HIGHS', None)),
        ('ECOS', getattr(cp, 'ECOS', None)),
        ('OSQP', getattr(cp, 'OSQP', None)),
        ('SCS', getattr(cp, 'SCS', None)),
    ]
    last_error = None
    for solver_name, solver_const in solver_attempts:
        if solver_const is None:
            continue
        try:
            problem.solve(solver=solver_const, verbose=False)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return solver_name
        except Exception as exc:  # pragma: no cover
            last_error = exc
    detail = f'Final solver status: {problem.status}.'
    if last_error is not None:
        detail += f' Last solver error: {last_error}'
    raise RuntimeError('CVXPY solve failed. ' + detail)


def optimal_dispatch_cvxpy(
    prices_gbp_per_kwh: np.ndarray,
    battery: BatteryParams,
    *,
    terminal_target_kwh: float | None = None,
    tiny_throughput_penalty: float = 1e-7,
) -> pd.DataFrame:
    if cp is None:
        raise ImportError('cvxpy is not installed. Install locally with: pip install cvxpy') from _CVXPY_IMPORT_ERROR
    if terminal_target_kwh is None:
        terminal_target_kwh = battery.initial_soc_kwh

    T = len(prices_gbp_per_kwh)
    dt = battery.dt_hours
    price = np.asarray(prices_gbp_per_kwh, dtype=float)

    charge = cp.Variable(T, nonneg=True, name='charge_kw')
    discharge = cp.Variable(T, nonneg=True, name='discharge_kw')
    energy = cp.Variable(T + 1, name='soc_kwh')

    constraints = [
        energy[0] == battery.initial_soc_kwh,
        energy[T] == float(terminal_target_kwh),
        energy >= 0,
        energy <= battery.energy_capacity_kwh,
        charge <= battery.max_charge_power_kw,
        discharge <= battery.max_discharge_power_kw,
    ]
    for t in range(T):
        constraints.append(
            energy[t + 1] == energy[t] + battery.eta_ch * charge[t] * dt - discharge[t] * dt / battery.eta_dis
        )

    purchase_cost = cp.sum(cp.multiply(price, charge * dt))
    revenue = cp.sum(cp.multiply(price, discharge * dt))
    throughput_penalty = tiny_throughput_penalty * cp.sum((charge + discharge) * dt)

    net_electricity_cost = purchase_cost - revenue
    problem = cp.Problem(cp.Minimize(net_electricity_cost + throughput_penalty), constraints)
    used_solver = _solve_cvxpy_problem(problem)

    charge_kw = np.asarray(charge.value, dtype=float).reshape(-1)
    discharge_kw = np.asarray(discharge.value, dtype=float).reshape(-1)
    soc = np.asarray(energy.value, dtype=float).reshape(-1)
    charge_kw[np.abs(charge_kw) < 1e-9] = 0.0
    discharge_kw[np.abs(discharge_kw) < 1e-9] = 0.0
    soc[np.abs(soc) < 1e-9] = 0.0

    return pd.DataFrame({
        'charge_kw': charge_kw,
        'discharge_kw': discharge_kw,
        'soc_start_kwh': soc[:-1],
        'soc_end_kwh': soc[1:],
        'action': np.where(charge_kw > 1e-6, 'CHARGE', np.where(discharge_kw > 1e-6, 'DISCHARGE', 'IDLE')),
        'reason': np.where(
            charge_kw > 1e-6,
            f'CVXPY minimum-net-cost dispatch ({used_solver})',
            np.where(discharge_kw > 1e-6, f'CVXPY minimum-net-cost dispatch ({used_solver})', f'no action ({used_solver})')
        ),
    })


def optimal_dispatch_cvxpy_carbon_aware(
    prices_gbp_per_kwh: np.ndarray,
    carbon_kg_per_kwh: np.ndarray,
    battery: BatteryParams,
    *,
    carbon_price_gbp_per_kg: float = 0.05,
    terminal_target_kwh: float | None = None,
    tiny_throughput_penalty: float = 1e-7,
    
) -> pd.DataFrame:
    if cp is None:
        raise ImportError('cvxpy is not installed. Install locally with: pip install cvxpy') from _CVXPY_IMPORT_ERROR
    if terminal_target_kwh is None:
        terminal_target_kwh = battery.initial_soc_kwh

    T = len(prices_gbp_per_kwh)
    dt = battery.dt_hours
    price = np.asarray(prices_gbp_per_kwh, dtype=float)
    carbon = np.asarray(carbon_kg_per_kwh, dtype=float)

    charge = cp.Variable(T, nonneg=True, name='charge_kw')
    discharge = cp.Variable(T, nonneg=True, name='discharge_kw')
    energy = cp.Variable(T + 1, name='soc_kwh')

    constraints = [
        energy[0] == battery.initial_soc_kwh,
        energy[T] == float(terminal_target_kwh),
        energy >= 0,
        energy <= battery.energy_capacity_kwh,
        charge <= battery.max_charge_power_kw,
        discharge <= battery.max_discharge_power_kw,
    ]
    for t in range(T):
        constraints.append(
            energy[t + 1] == energy[t] + battery.eta_ch * charge[t] * dt - discharge[t] * dt / battery.eta_dis
        )

    purchase_cost = cp.sum(cp.multiply(price, charge * dt))
    revenue = cp.sum(cp.multiply(price, discharge * dt))
    net_carbon_kg = cp.sum(cp.multiply(carbon, (charge - discharge) * dt))
    throughput_penalty = tiny_throughput_penalty * cp.sum((charge + discharge) * dt)

    # Carbon-aware extension written in the same minimisation style
    net_electricity_cost = purchase_cost - revenue
    carbon_penalty = carbon_price_gbp_per_kg * net_carbon_kg
    problem = cp.Problem(cp.Minimize(net_electricity_cost + carbon_penalty + throughput_penalty), constraints)
    used_solver = _solve_cvxpy_problem(problem)

    charge_kw = np.asarray(charge.value, dtype=float).reshape(-1)
    discharge_kw = np.asarray(discharge.value, dtype=float).reshape(-1)
    soc = np.asarray(energy.value, dtype=float).reshape(-1)
    charge_kw[np.abs(charge_kw) < 1e-9] = 0.0
    discharge_kw[np.abs(discharge_kw) < 1e-9] = 0.0
    soc[np.abs(soc) < 1e-9] = 0.0

    return pd.DataFrame({
        'charge_kw': charge_kw,
        'discharge_kw': discharge_kw,
        'soc_start_kwh': soc[:-1],
        'soc_end_kwh': soc[1:],
        'action': np.where(charge_kw > 1e-6, 'CHARGE', np.where(discharge_kw > 1e-6, 'DISCHARGE', 'IDLE')),
        'reason': np.where(
            charge_kw > 1e-6,
            f'CVXPY carbon-aware minimum-net-cost dispatch ({used_solver})',
            np.where(discharge_kw > 1e-6, f'CVXPY carbon-aware minimum-net-cost dispatch ({used_solver})', f'no action ({used_solver})')
        ),
    })


def attach_market_columns(
    base_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    method_name: str,
    battery: BatteryParams,
    *,
    carbon_price_gbp_per_kg: float = 0.0,
) -> pd.DataFrame:
    out = pd.concat([base_df.reset_index(drop=True), schedule_df.reset_index(drop=True)], axis=1)
    dt = battery.dt_hours
    price = out['day_ahead_price_gbp_per_kwh']
    carbon = out['carbon_intensity_kg_per_kwh']
    out['charge_energy_kwh'] = out['charge_kw'] * dt
    out['discharge_energy_kwh'] = out['discharge_kw'] * dt
    out['purchase_cost_gbp'] = out['charge_energy_kwh'] * price
    out['sale_revenue_gbp'] = out['discharge_energy_kwh'] * price
    out['step_profit_gbp'] = out['sale_revenue_gbp'] - out['purchase_cost_gbp']
    out['cumulative_profit_gbp'] = out['step_profit_gbp'].cumsum()
    out['charge_emissions_kg'] = out['charge_energy_kwh'] * carbon
    out['discharge_displacement_kg'] = out['discharge_energy_kwh'] * carbon
    out['step_net_carbon_kg'] = out['charge_emissions_kg'] - out['discharge_displacement_kg']
    out['cumulative_net_carbon_kg'] = out['step_net_carbon_kg'].cumsum()
    out['carbon_cost_gbp_equiv'] = out['step_net_carbon_kg'] * carbon_price_gbp_per_kg
    out['step_objective_gbp_equiv'] = out['step_profit_gbp'] - out['carbon_cost_gbp_equiv']
    out['method'] = method_name
    return out


def compute_metrics(df: pd.DataFrame, method_name: str, battery: BatteryParams, *, carbon_price_gbp_per_kg: float = 0.0) -> DispatchMetrics:
    charged = float(df['charge_energy_kwh'].sum())
    discharged = float(df['discharge_energy_kwh'].sum())
    revenue = float(df['sale_revenue_gbp'].sum())
    purchase_cost = float(df['purchase_cost_gbp'].sum())
    final_soc = float(df['soc_end_kwh'].iloc[-1])
    throughput = charged + discharged
    eq_cycles = discharged / battery.energy_capacity_kwh if battery.energy_capacity_kwh > 0 else math.nan
    net_carbon = float(df['step_net_carbon_kg'].sum())
    objective_value = (revenue - purchase_cost) - carbon_price_gbp_per_kg * net_carbon
    return DispatchMetrics(
        method=method_name,
        total_profit_gbp=revenue - purchase_cost,
        gross_revenue_gbp=revenue,
        gross_purchase_cost_gbp=purchase_cost,
        charged_energy_kwh=charged,
        discharged_energy_kwh=discharged,
        total_throughput_kwh=throughput,
        final_soc_kwh=final_soc,
        soc_min_kwh=float(df['soc_end_kwh'].min()),
        soc_max_kwh=float(df['soc_end_kwh'].max()),
        equivalent_full_cycles=eq_cycles,
        terminal_soc_gap_kwh=abs(final_soc - battery.initial_soc_kwh),
        net_carbon_kg=net_carbon,
        objective_value_gbp_equiv=objective_value,
    )


def verification_metrics(df: pd.DataFrame, battery: BatteryParams, *, terminal_target_kwh: float | None = None) -> Dict[str, float]:
    if terminal_target_kwh is None:
        terminal_target_kwh = battery.initial_soc_kwh
    price_rebuilt = df['day_ahead_price_gbp_per_mwh'] / 1000.0
    soc_residual = (
        df['soc_end_kwh'] - df['soc_start_kwh'] - battery.eta_ch * df['charge_kw'] * battery.dt_hours + df['discharge_kw'] * battery.dt_hours / battery.eta_dis
    )
    simultaneous_kw = np.minimum(df['charge_kw'].to_numpy(), df['discharge_kw'].to_numpy())
    reconstructed_profit = float(df['sale_revenue_gbp'].sum() - df['purchase_cost_gbp'].sum())
    direct_profit = float(df['step_profit_gbp'].sum())
    return {
        'unit_conversion_max_abs_error': float(np.max(np.abs(df['day_ahead_price_gbp_per_kwh'] - price_rebuilt))),
        'soc_dynamics_max_abs_error': float(np.max(np.abs(soc_residual))),
        'soc_min_kwh': float(df['soc_end_kwh'].min()),
        'soc_max_kwh': float(df['soc_end_kwh'].max()),
        'charge_max_kw': float(df['charge_kw'].max()),
        'discharge_max_kw': float(df['discharge_kw'].max()),
        'trade_energy_min_kwh': float(min(df['charge_energy_kwh'].min(), df['discharge_energy_kwh'].min())),
        'profit_reconstruction_abs_error_gbp': abs(reconstructed_profit - direct_profit),
        'simultaneous_charge_discharge_max_kw': float(np.max(simultaneous_kw)),
        'terminal_soc_gap_kwh': abs(float(df['soc_end_kwh'].iloc[-1]) - float(terminal_target_kwh)),
    }


def verification_to_frame(metrics: Dict[str, float], method_name: str, battery: BatteryParams) -> pd.DataFrame:
    criteria = {
        'unit_conversion_max_abs_error': '<= 1e-12',
        'soc_dynamics_max_abs_error': '<= 1e-6',
        'soc_min_kwh': f'>= 0 and <= {battery.energy_capacity_kwh:.6f}',
        'soc_max_kwh': f'>= 0 and <= {battery.energy_capacity_kwh:.6f}',
        'charge_max_kw': f'<= {battery.max_charge_power_kw:.6f}',
        'discharge_max_kw': f'<= {battery.max_discharge_power_kw:.6f}',
        'trade_energy_min_kwh': '>= 0',
        'profit_reconstruction_abs_error_gbp': '<= 1e-6',
        'simultaneous_charge_discharge_max_kw': '<= 1e-6',
        'terminal_soc_gap_kwh': '<= 1e-6',
    }
    return pd.DataFrame({
        'method': method_name,
        'metric': list(metrics.keys()),
        'value': list(metrics.values()),
        'criterion': [criteria[k] for k in metrics.keys()],
    })


def _choose_zoom_window(df: pd.DataFrame, window_hours: int = 24 * 7) -> slice:
    prices = df['day_ahead_price_gbp_per_kwh'].to_numpy(dtype=float)
    T = len(prices)
    if T <= window_hours:
        return slice(0, T)
    best_start = 0
    best_score = -np.inf
    for start in range(0, T - window_hours + 1, 24):
        chunk = prices[start:start + window_hours]
        score = float(np.max(chunk) - np.min(chunk))
        if score > best_score:
            best_score = score
            best_start = start
    return slice(best_start, best_start + window_hours)


def plot_mainline_price_and_dispatch_zoom(simple_df: pd.DataFrame, optimal_df: pd.DataFrame, outpath: Path) -> None:
    zoom = _choose_zoom_window(optimal_df)
    s = simple_df.iloc[zoom]
    o = optimal_df.iloc[zoom]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(o['timestamp'], o['day_ahead_price_gbp_per_kwh'])
    axes[0].set_ylabel('Price (£/kWh)')
    axes[0].set_title('Case B mainline - price and dispatch (zoom window)')

    axes[1].step(s['timestamp'], s['charge_kw'], where='mid', label='Charge')
    axes[1].step(s['timestamp'], -s['discharge_kw'], where='mid', label='Discharge')
    axes[1].set_ylabel('Simple rule (kW)')
    axes[1].legend(loc='upper right')

    axes[2].step(o['timestamp'], o['charge_kw'], where='mid', label='Charge')
    axes[2].step(o['timestamp'], -o['discharge_kw'], where='mid', label='Discharge')
    axes[2].set_ylabel('Optimal (kW)')
    axes[2].set_xlabel('Timestamp')
    axes[2].legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_mainline_soc_comparison(simple_df: pd.DataFrame, optimal_df: pd.DataFrame, battery: BatteryParams, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(simple_df['timestamp'], simple_df['soc_end_kwh'], label='Simple rule')
    ax.plot(optimal_df['timestamp'], optimal_df['soc_end_kwh'], label='Optimal')
    ax.axhline(battery.initial_soc_kwh, linestyle='--', linewidth=1.0, label='Initial / terminal target')
    ax.set_title('Case B mainline - state of charge comparison')
    ax.set_ylabel('SOC (kWh)')
    ax.set_xlabel('Timestamp')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_mainline_cumulative_profit(simple_df: pd.DataFrame, optimal_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(simple_df['timestamp'], simple_df['cumulative_profit_gbp'], label='Simple rule')
    ax.plot(optimal_df['timestamp'], optimal_df['cumulative_profit_gbp'], label='Optimal')
    ax.set_title('Case B mainline - cumulative arbitrage profit')
    ax.set_ylabel('Profit (£)')
    ax.set_xlabel('Timestamp')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_extension_profit_vs_carbon_zoom(profit_df: pd.DataFrame, carbon_df: pd.DataFrame, outpath: Path) -> None:
    zoom = _choose_zoom_window(profit_df)
    p = profit_df.iloc[zoom]
    c = carbon_df.iloc[zoom]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(p['timestamp'], p['day_ahead_price_gbp_per_kwh'], label='Price')
    ax0b = axes[0].twinx()
    ax0b.plot(p['timestamp'], p['carbon_intensity_kg_per_kwh'], alpha=0.6, label='Carbon intensity')
    axes[0].set_ylabel('Price (£/kWh)')
    ax0b.set_ylabel('Carbon (kg/kWh)')
    axes[0].set_title('Case B extension - profit-only vs carbon-aware (zoom window)')

    axes[1].step(p['timestamp'], p['charge_kw'], where='mid', label='Charge')
    axes[1].step(p['timestamp'], -p['discharge_kw'], where='mid', label='Discharge')
    axes[1].set_ylabel('Profit-only (kW)')
    axes[1].legend(loc='upper right')

    axes[2].step(c['timestamp'], c['charge_kw'], where='mid', label='Charge')
    axes[2].step(c['timestamp'], -c['discharge_kw'], where='mid', label='Discharge')
    axes[2].set_ylabel('Carbon-aware (kW)')
    axes[2].legend(loc='upper right')

    axes[3].plot(p['timestamp'], p['cumulative_profit_gbp'], label='Profit-only cumulative profit')
    axes[3].plot(c['timestamp'], c['cumulative_profit_gbp'], label='Carbon-aware cumulative profit')
    axes[3].set_ylabel('Profit (£)')
    axes[3].set_xlabel('Timestamp')
    axes[3].legend(loc='best')

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_extension_tradeoff(profit_metrics: DispatchMetrics, carbon_metrics: DispatchMetrics, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    labels = ['Profit-only', 'Carbon-aware']
    profits = [profit_metrics.total_profit_gbp, carbon_metrics.total_profit_gbp]
    carbons = [profit_metrics.net_carbon_kg, carbon_metrics.net_carbon_kg]
    axes[0].bar(labels, profits)
    axes[0].set_title('Financial profit')
    axes[0].set_ylabel('£')
    axes[1].bar(labels, carbons)
    axes[1].set_title('Net carbon')
    axes[1].set_ylabel('kg CO2')
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / 'caseB_grid_battery_market_hourly.csv'

    parser = argparse.ArgumentParser(description='EGS Case B - mainline and carbon-aware extension')
    parser.add_argument('--input', type=Path, default=default_input, help='Path to caseB_grid_battery_market_hourly.csv')
    parser.add_argument('--outdir', type=Path, default=Path('caseB_outputs_main_and_extension'), help='Output directory')
    parser.add_argument('--carbon-price', type=float, default=0.05, help='Carbon weight in GBP per kg CO2')
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    battery = BatteryParams()
    base = load_case_b(args.input)
    prices = base['day_ahead_price_gbp_per_kwh'].to_numpy(dtype=float)
    carbon = base['carbon_intensity_kg_per_kwh'].to_numpy(dtype=float)

    simple_sched = simple_rule_dispatch(base, battery, terminal_target_kwh=battery.initial_soc_kwh)
    profit_sched = optimal_dispatch_cvxpy(prices, battery, terminal_target_kwh=battery.initial_soc_kwh)
    carbon_sched = optimal_dispatch_cvxpy_carbon_aware(prices, carbon, battery, carbon_price_gbp_per_kg=args.carbon_price, terminal_target_kwh=battery.initial_soc_kwh)

    simple_df = attach_market_columns(base, simple_sched, 'simple_rule_simulation', battery, carbon_price_gbp_per_kg=0.0)
    profit_df = attach_market_columns(base, profit_sched, 'optimal_profit_only', battery, carbon_price_gbp_per_kg=0.0)
    carbon_df = attach_market_columns(base, carbon_sched, 'optimal_carbon_aware', battery, carbon_price_gbp_per_kg=args.carbon_price)

    simple_metrics = compute_metrics(simple_df, 'simple_rule_simulation', battery, carbon_price_gbp_per_kg=0.0)
    profit_metrics = compute_metrics(profit_df, 'optimal_profit_only', battery, carbon_price_gbp_per_kg=0.0)
    carbon_metrics = compute_metrics(carbon_df, 'optimal_carbon_aware', battery, carbon_price_gbp_per_kg=args.carbon_price)

    simple_ver = verification_metrics(simple_df, battery)
    profit_ver = verification_metrics(profit_df, battery)
    carbon_ver = verification_metrics(carbon_df, battery)

    simple_df.to_csv(args.outdir / 'caseB_hourly_dispatch_simple_rule.csv', index=False)
    profit_df.to_csv(args.outdir / 'caseB_hourly_dispatch_optimal_profit_only.csv', index=False)
    carbon_df.to_csv(args.outdir / 'caseB_hourly_dispatch_optimal_carbon_aware.csv', index=False)

    pd.DataFrame([asdict(simple_metrics), asdict(profit_metrics), asdict(carbon_metrics)]).to_csv(args.outdir / 'caseB_summary_metrics.csv', index=False)
    pd.concat([
        verification_to_frame(simple_ver, 'simple_rule_simulation', battery),
        verification_to_frame(profit_ver, 'optimal_profit_only', battery),
        verification_to_frame(carbon_ver, 'optimal_carbon_aware', battery),
    ], ignore_index=True).to_csv(args.outdir / 'caseB_verification_checks.csv', index=False)

    plot_mainline_price_and_dispatch_zoom(simple_df, profit_df, args.outdir / 'caseB_mainline_price_and_dispatch_zoom.png')
    plot_mainline_soc_comparison(simple_df, profit_df, battery, args.outdir / 'caseB_mainline_soc_comparison.png')
    plot_mainline_cumulative_profit(simple_df, profit_df, args.outdir / 'caseB_mainline_cumulative_profit.png')
    plot_extension_profit_vs_carbon_zoom(profit_df, carbon_df, args.outdir / 'caseB_extension_profit_vs_carbon_zoom.png')
    plot_extension_tradeoff(profit_metrics, carbon_metrics, args.outdir / 'caseB_extension_tradeoff.png')

    summary = {
        'input_file': args.input.name,
        'battery_parameters': asdict(battery),
        'mainline_methods': ['simple_rule_simulation', 'optimal_profit_only'],
        'extension_method': 'optimal_carbon_aware',
        'carbon_extension': {
            'extension_name': 'Carbon-aware dispatch',
            'method': 'same LP structure written as minimum net electricity cost plus a carbon penalty',
            'carbon_price_gbp_per_kg': float(args.carbon_price),
            'net_carbon_definition': 'charge emissions minus discharge displacement',
        },
        'unit_check_example': {
            'first_price_gbp_per_mwh': float(base['day_ahead_price_gbp_per_mwh'].iloc[0]),
            'first_price_gbp_per_kwh': float(base['day_ahead_price_gbp_per_kwh'].iloc[0]),
            'explanation': 'GBP/MWh divided by 1000 equals GBP/kWh',
        },
        'summary_metrics': [asdict(simple_metrics), asdict(profit_metrics), asdict(carbon_metrics)],
        'verification_metrics': {
            'simple_rule_simulation': simple_ver,
            'optimal_profit_only': profit_ver,
            'optimal_carbon_aware': carbon_ver,
        },
        'notes': {
            'base_case_scope': 'included',
            'extension_scope': 'included',
            'ancillary_column_use': 'not used in the model',
            'imbalance_column_use': 'not used in the model',
            'carbon_column_use': 'used only in the extension objective',
            'simple_rule_method': 'fixed daily charging and discharging windows without look-ahead optimisation',
            'optimal_solver_family': 'CVXPY linear programming with teacher-style minimum-net-cost objective',
            'execution_note': 'Local cvxpy installation is required.',
        },
    }
    with open(args.outdir / 'caseB_run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Data export completed')


if __name__ == '__main__':
    main()
