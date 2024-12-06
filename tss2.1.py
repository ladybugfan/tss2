import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, Function, dsolve, Eq, exp, sin, lambdify, simplify


def solve_analytically(m, k, h, f_symbolic, x0, v0):
    t = symbols('t', real=True)
    x = Function('x')(t)
    eq = Eq(m * x.diff(t, t) + h * x.diff(t) + k * x, f_symbolic)
    ics = {x.subs(t, 0): x0, x.diff(t).subs(t, 0): v0}
    sol = dsolve(eq, x, ics=ics)
    return lambdify(t, simplify(sol.rhs), modules=['numpy'])

def solve_numerically(m, k, h, f_numeric, time_interval, time_array, x0, v0):
    def ode_system(t, y):
        x1, x2 = y
        return [x2, (f_numeric(t) - h * x2 - k * x1) / m]

    y0 = [x0, v0]
    sol = solve_ivp(ode_system, time_interval, y0, t_eval=time_array)
    return sol.y[0]

def plot_results(forces, h_values, m, k, x0, v0, time_interval, time_array):
    for name, f_numeric, f_symbolic in forces:
        for h_type, h in h_values.items():
            analytical = solve_analytically(m, k, h, f_symbolic, x0, v0)
            numerical = solve_numerically(m, k, h, f_numeric, time_interval, time_array, x0, v0)
            plt.figure(figsize=(8, 4))
            plt.plot(time_array, analytical(time_array), label="Analytical", color="blue")
            plt.plot(time_array, numerical, linestyle="--", label="Numerical", color="red")
            plt.title(f"{h_type} : {name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Displacement (m)")
            plt.legend()
            plt.tight_layout()
            plt.show()

def main():
    m, k = 4.0, 1.0
    x0, v0 = 0.0, 1.0
    time_interval = (0, 30)
    time_array = np.linspace(time_interval[0], time_interval[1], 400)
    h_values = {"h^2 < 4km": 3.0, "h^2 > 4km": 17.0}
    forces = [
        ("f(t) = 0", lambda t: 0, 0),
        ("f(t) = t - 1", lambda t: t - 1, symbols('t') - 1),
        ("f(t) = e^(-t)", lambda t: np.exp(-t), exp(-symbols('t'))),
        ("f(t) = sin(t)", lambda t: np.sin(t), sin(symbols('t')))
    ]
    plot_results(forces, h_values, m, k, x0, v0, time_interval, time_array)

if __name__ == "__main__":
    main()
