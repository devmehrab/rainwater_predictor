"""
simulation.py
Defines the rainwater harvesting mathematical equations, runs simulations,
and generates the Matplotlib comparison plot.
"""

import math
import matplotlib.pyplot as plt
from numerical_methods import (
    euler_method,
    modified_euler_method,
    bisection_method,
    newton_raphson_method,
)


def run_accumulation_simulation(area, intensity, efficiency, v_initial, duration, step):
    """
    Simulates water accumulation over time using Euler and Modified Euler methods.
    ODE: dV/dt = (A * R * C) / 1000
    """
    # Define rate function f(t, V)
    def rate_of_collection(t, v):
        return (area * intensity * efficiency) / 1000.0

    num_steps = math.ceil(duration / step)

    t_euler, v_euler = euler_method(rate_of_collection, 0.0, v_initial, step, num_steps)
    t_mod, v_mod = modified_euler_method(rate_of_collection, 0.0, v_initial, step, num_steps)

    return t_euler, v_euler, v_mod


def estimate_filling_time(area, intensity, efficiency, v_initial, v_capacity, max_search_time=10000.0):
    """
    Finds the time t required to fill the tank using the Bisection Method.
    Equation: f(t) = V_initial + ((A * R * C) / 1000) * t - V_capacity = 0
    """
    if v_initial >= v_capacity:
        return 0.0, []

    inflow_rate = (area * intensity * efficiency) / 1000.0
    if inflow_rate <= 0:
        return None, []

    def f_time(t):
        return v_initial + inflow_rate * t - v_capacity

    # Automatically expand upper bracket if needed so root is guaranteed inside [0, b]
    a = 0.0
    b = max(10.0, max_search_time)
    if f_time(b) < 0:
        return None, []

    # Tighten upper bracket to a reasonable scale for cleaner iteration tables
    while f_time(b / 2.0) > 0 and b > 1.0:
        b /= 2.0

    return bisection_method(f_time, a, b)


def estimate_required_intensity(area, efficiency, v_initial, v_capacity, target_duration):
    """
    Calculates required rainfall intensity R to fill the tank in target_duration (T)
    using the Newton-Raphson Method.
    Equation:   f(R)  = V_initial + ((A * R * C) / 1000) * T - V_capacity = 0
    Derivative: f'(R) = (A * C * T) / 1000
    """
    if v_initial >= v_capacity:
        return 0.0, []

    def f_intensity(r):
        return v_initial + ((area * r * efficiency) / 1000.0) * target_duration - v_capacity

    def df_intensity(r):
        return (area * efficiency * target_duration) / 1000.0

    # Initial guess R0 = 10.0 mm/hr
    return newton_raphson_method(f_intensity, df_intensity, r0=10.0)


def plot_simulation(times, v_euler, v_mod, v_capacity):
    """
    Plots Euler vs. Modified Euler water accumulation against tank capacity.
    """
    plt.figure(figsize=(9, 5))

    # Plot Euler Method
    plt.plot(
        times,
        v_euler,
        label="Euler Method",
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#1f77b4",
    )

    # Plot Modified Euler Method
    plt.plot(
        times,
        v_mod,
        label="Modified Euler Method (Heun's)",
        marker="s",
        linestyle="--",
        linewidth=2,
        color="#ff7f0e",
    )

    # Plot Horizontal Tank Capacity Line
    plt.axhline(
        y=v_capacity,
        color="#d62728",
        linestyle=":",
        linewidth=2,
        label=f"Tank Capacity ({v_capacity:.2f} L)",
    )

    # Highlight overflow point if tank reaches capacity
    for t, v in zip(times, v_mod):
        if v >= v_capacity:
            plt.scatter([t], [v_capacity], color="#d62728", zorder=5, s=80)
            plt.annotate(
                f"Capacity Reached (~{t:.2f} hrs)",
                xy=(t, v_capacity),
                xytext=(t, v_capacity * 0.88),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                fontsize=9,
                fontweight="bold",
                color="#d62728",
            )
            break

    plt.title("Rainwater Harvesting Tank Accumulation Simulation", fontsize=13, fontweight="bold")
    plt.xlabel("Time (hours)", fontsize=11)
    plt.ylabel("Water Volume (litres)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()