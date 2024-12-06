import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def create_grid(start, end, num_points):
    return np.linspace(start, end, num_points)

def define_system(x, y):
    u, v = y
    du_dx = v
    dv_dx = x**2 * v + (2 / x**2) * u + 1 + (4 / x**2)
    return np.vstack([du_dx, dv_dx])

def boundary_conditions(ya, yb):
    bc1 = 2 * ya[0] - ya[1] - 6
    bc2 = yb[0] + 3 * yb[1] + 1
    return np.array([bc1, bc2])

def initial_guess_for_solution(x):
    return np.zeros((2, x.size))

def compute_second_derivative(x, solution):
    return define_system(x, solution)[1]

def plot_solution(x, y, y_prime, y_double_prime):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(x, y, label='y(x)', color='b')
    axes[0].set_title('Plot of the function y(x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y(x)')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x, y_prime, label="y'(x)", color='r')
    axes[1].set_title("Plot of the first derivative y'(x)")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("y'(x)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(x, y_double_prime, label="y''(x)", color='g')
    axes[2].set_title("Plot of the second derivative y''(x)")
    axes[2].set_xlabel('x')
    axes[2].set_ylabel("y''(x)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def display_table(solution, x_vals):
    y_vals = solution.sol(x_vals)[0]
    y_prime_vals = solution.sol(x_vals)[1]
    y_double_prime_vals = define_system(x_vals, solution.sol(x_vals))[1]
    
    print("Values of the function and its derivatives at selected points:")
    print("  x       y(x)       y'(x)       y''(x)")
    for xi, yi, ypi, ydpi in zip(x_vals, y_vals, y_prime_vals, y_double_prime_vals):
        print(f"{xi:.3f}    {yi:.6f}    {ypi:.6f}    {ydpi:.6f}")

def main():
    x_range = create_grid(0.5, 1, 100)
    initial_guess = initial_guess_for_solution(x_range)
    
    solution = solve_bvp(define_system, boundary_conditions, x_range, initial_guess)
    
    x_solution = solution.x
    y_solution = solution.y[0]
    y_prime_solution = solution.y[1]
    y_double_prime_solution = compute_second_derivative(x_solution, solution.y)
    
    plot_solution(x_solution, y_solution, y_prime_solution, y_double_prime_solution)
    
    x_for_table = np.linspace(0.5, 1, 10)
    display_table(solution, x_for_table)

if __name__ == "__main__":
    main()
