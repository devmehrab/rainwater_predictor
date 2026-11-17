"""
numerical_methods.py
Manual implementations of Euler, Modified Euler (Heun's), Bisection,
and Newton-Raphson methods for the Rainwater Harvesting Capacity Predictor.
"""


def euler_method(f, t0, v0, h, steps):
    """
    Solves an ODE dv/dt = f(t, v) using the explicit Euler Method.
    Formula: V_{n+1} = V_n + h * f(t_n, V_n)
    """
    t_values = [t0]
    v_values = [v0]

    t = t0
    v = v0

    for _ in range(steps):
        # Compute slope at the beginning of the interval
        slope = f(t, v)
        # Step forward using Euler's linear approximation
        v_next = v + h * slope
        t_next = round(t + h, 10)  # Avoid floating-point drift

        t_values.append(t_next)
        v_values.append(v_next)

        t = t_next
        v = v_next

    return t_values, v_values


def modified_euler_method(f, t0, v0, h, steps):
    """
    Solves an ODE dv/dt = f(t, v) using Modified Euler (Heun's Method).
    Predictor: V_p = V_n + h * f(t_n, V_n)
    Corrector: V_{n+1} = V_n + (h / 2) * [f(t_n, V_n) + f(t_{n+1}, V_p)]
    """
    t_values = [t0]
    v_values = [v0]

    t = t0
    v = v0

    for _ in range(steps):
        t_next = round(t + h, 10)

        # 1. Predictor step: estimate next volume using initial slope
        k1 = f(t, v)
        v_predict = v + h * k1

        # 2. Corrector step: average initial slope and predicted end slope
        k2 = f(t_next, v_predict)
        v_next = v + (h / 2.0) * (k1 + k2)

        t_values.append(t_next)
        v_values.append(v_next)

        t = t_next
        v = v_next

    return t_values, v_values


def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    """
    Finds a root of f(t) = 0 in interval [a, b] using the Bisection Method.
    Returns (root, iteration_history) or (None, []) if root is not bracketed.
    """
    fa = f(a)
    fb = f(b)

    # Check if root is already at an endpoint
    if abs(fa) <= tol:
        return a, []
    if abs(fb) <= tol:
        return b, []

    # Intermediate Value Theorem check: f(a) and f(b) must have opposite signs
    if fa * fb > 0:
        return None, []

    history = []
    c_prev = a

    for iteration in range(1, max_iter + 1):
        # Compute midpoint
        c = (a + b) / 2.0
        fc = f(c)
        abs_error = abs(c - c_prev) if iteration > 1 else abs(b - a) / 2.0

        history.append({
            "iter": iteration,
            "a": a,
            "b": b,
            "midpoint": c,
            "f_mid": fc,
            "error": abs_error
        })

        # Check convergence criteria
        if abs(fc) <= tol or abs_error <= tol:
            return c, history

        # Narrow the bracket to the sub-interval containing the sign change
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_prev = c

    return (a + b) / 2.0, history


def newton_raphson_method(f, df, r0, tol=1e-6, max_iter=100):
    """
    Finds a root of f(R) = 0 using the Newton-Raphson Method.
    Formula: R_{n+1} = R_n - f(R_n) / f'(R_n)
    Returns (root, iteration_history) or (None, history) on zero derivative.
    """
    r = r0
    history = []

    for iteration in range(1, max_iter + 1):
        fr = f(r)
        dfr = df(r)

        # Division-by-zero protection
        if abs(dfr) < 1e-12:
            return None, history

        r_next = r - (fr / dfr)
        abs_error = abs(r_next - r)

        history.append({
            "iter": iteration,
            "estimate": r,
            "f_val": fr,
            "next_estimate": r_next,
            "error": abs_error
        })

        # Check convergence criteria
        if abs(fr) <= tol or abs_error <= tol:
            return r_next, history

        r = r_next

    return r, history