"""
main.py
Entry point for the Rainwater Harvesting Capacity Predictor.
Collects validated user inputs, executes numerical simulations, and displays results.
"""

from simulation import (
    run_accumulation_simulation,
    estimate_filling_time,
    estimate_required_intensity,
    plot_simulation,
)


def get_validated_float(prompt, default, min_val=None, max_val=None, allow_zero=True):
    """Prompts user for a float with a default fallback and range validation."""
    while True:
        raw = input(f"{prompt} [Default: {default}]: ").strip()
        if raw == "":
            val = float(default)
        else:
            try:
                val = float(raw)
            except ValueError:
                print("  Invalid input. Please enter a numeric value.")
                continue

        if not allow_zero and val == 0:
            print("  Value cannot be zero.")
            continue
        if min_val is not None and val < min_val:
            print(f"  Value must be at least {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"  Value must be at most {max_val}.")
            continue

        return val


def main():
    print("=" * 68)
    print("        RAINWATER HARVESTING CAPACITY PREDICTOR")
    print("        Numerical Methods Academic Demonstration")
    print("=" * 68)
    print("Press ENTER at any prompt to accept the default value.\n")

    # 1. Gather and Validate Inputs
    area = get_validated_float("Rooftop area (m²)", default=120.0, min_val=0.01, allow_zero=False)
    capacity = get_validated_float("Storage tank capacity (litres)", default=50.0, min_val=0.01, allow_zero=False)
    v_initial = get_validated_float(
        "Initial water volume (litres)", default=5.0, min_val=0.0, max_val=capacity
    )
    intensity = get_validated_float("Rainfall intensity (mm/hour)", default=25.0, min_val=0.0)
    eff_percent = get_validated_float("Collection efficiency (%)", default=85.0, min_val=0.0, max_val=100.0)
    duration = get_validated_float("Simulation duration (hours)", default=12.0, min_val=0.1, allow_zero=False)
    step = get_validated_float(
        "Time step h (hours)", default=1.0, min_val=0.01, max_val=duration, allow_zero=False
    )
    desired_time = get_validated_float(
        "Desired filling duration T (hours)", default=10.0, min_val=0.01, allow_zero=False
    )

    efficiency = eff_percent / 100.0

    # 2. Method 1 & Method 2: Water Accumulation Simulation
    times, v_euler, v_mod = run_accumulation_simulation(
        area, intensity, efficiency, v_initial, duration, step
    )

    print("\n" + "=" * 68)
    print("A. WATER ACCUMULATION SIMULATION (EULER vs. MODIFIED EULER)")
    print("=" * 68)
    print(f"{'Time (hrs)':<14} | {'Euler Method (L)':<22} | {'Modified Euler (L)':<22}")
    print("-" * 68)
    for t, ve, vm in zip(times, v_euler, v_mod):
        print(f"{t:<14.2f} | {ve:<22.6f} | {vm:<22.6f}")

    # 3. Method 3: Tank Filling Time (Bisection Method)
    print("\n" + "=" * 68)
    print("B. TANK FILLING TIME (BISECTION METHOD)")
    print("=" * 68)
    fill_time, bisection_hist = estimate_filling_time(
        area, intensity, efficiency, v_initial, capacity
    )

    if v_initial >= capacity:
        print("The tank is already full at t = 0.00 hours.")
    elif fill_time is None:
        print("Tank cannot be filled within the searchable interval (zero inflow or huge capacity).")
    else:
        first_row = bisection_hist[0]
        print(f"Initial Interval: [{first_row['a']:.4f}, {first_row['b']:.4f}] hours\n")
        print(f"{'Iter':<6} | {'Midpoint (t)':<14} | {'f(Midpoint)':<16} | {'Abs Error':<14}")
        print("-" * 68)
        for row in bisection_hist:
            print(
                f"{row['iter']:<6} | {row['midpoint']:<14.6f} | "
                f"{row['f_mid']:<16.6e} | {row['error']:<14.6e}"
            )
        print(f"\n-> Final Estimated Filling Time: {fill_time:.4f} hours")

    # 4. Method 4: Required Rainfall Intensity (Newton-Raphson Method)
    print("\n" + "=" * 68)
    print(f"C. REQUIRED RAINFALL INTENSITY FOR T = {desired_time} HRS (NEWTON-RAPHSON)")
    print("=" * 68)
    req_intensity, nr_hist = estimate_required_intensity(
        area, efficiency, v_initial, capacity, desired_time
    )

    if v_initial >= capacity:
        print("Required Rainfall Intensity: 0.00 mm/hr (Tank is already full).")
    elif req_intensity is None:
        print("Newton-Raphson failed due to zero derivative (check area/efficiency).")
    else:
        print(f"{'Iter':<6} | {'Estimate R_n':<16} | {'f(R_n)':<16} | {'Abs Error':<14}")
        print("-" * 68)
        for row in nr_hist:
            print(
                f"{row['iter']:<6} | {row['estimate']:<16.6f} | "
                f"{row['f_val']:<16.6e} | {row['error']:<14.6e}"
            )
        print(f"\n-> Final Required Rainfall Intensity: {req_intensity:.4f} mm/hour")

    # 5. Practical Summary
    final_simulated_vol = v_mod[-1]
    collected_water = final_simulated_vol - v_initial
    overflowed = final_simulated_vol > capacity
    stored_water = min(final_simulated_vol, capacity)
    excess_water = max(0.0, final_simulated_vol - capacity)
    remaining_capacity = max(0.0, capacity - stored_water)

    print("\n" + "=" * 68)
    print("D. PRACTICAL SUMMARY")
    print("=" * 68)
    print(f"Initial Water Volume         : {v_initial:.4f} L")
    print(f"Estimated Collected Water    : {collected_water:.4f} L")
    print(f"Physically Stored Water      : {stored_water:.4f} L (Tank Cap: {capacity:.2f} L)")
    print(f"Tank Overflow Status         : {'YES - Overflow Occurred' if overflowed else 'NO - Within Capacity'}")
    if overflowed:
        print(f"Excess Water Lost (Overflow) : {excess_water:.4f} L")
    print(f"Remaining Storage Capacity   : {remaining_capacity:.4f} L")
    print(
        f"Estimated Time to Fill Tank  : "
        f"{f'{fill_time:.4f} hours' if fill_time is not None else 'N/A'}"
    )
    print(
        f"Required Rainfall Intensity  : "
        f"{f'{req_intensity:.4f} mm/hr (to fill in {desired_time} hrs)' if req_intensity is not None else 'N/A'}"
    )
    print("=" * 68)

    # 6. Generate Graph
    print("\nDisplaying Matplotlib simulation graph...")
    plot_simulation(times, v_euler, v_mod, capacity)


if __name__ == "__main__":
    main()