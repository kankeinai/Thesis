import casadi as ca
import numpy as np
import time 
from pathlib import Path

N = 10000
T = 1.0
dt = T / N
t = np.linspace(0, T, N)

def solve_linear():
    x = ca.MX.sym('x', N)
    u = ca.MX.sym('u', N)
    cost = 0.5 * dt * ca.sumsqr(x) + 0.5 * dt * ca.sumsqr(u)
    constraints = [x[0] - 1]
    for i in range(N - 1):
        dx = -x[i] + u[i]
        constraints.append(x[i+1] - (x[i] + dt * dx))
    g = ca.vertcat(*constraints)
    vars = ca.vertcat(x, u)
    prob = {'x': vars, 'f': cost, 'g': g}
    tic = time.time()
    solver = ca.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level': 0, 'print_time': 0})
    sol = solver(x0=np.zeros(2*N), lbg=0, ubg=0)
    toc = time.time()
    x_opt = np.array(sol['x'][:N]).reshape(-1)
    u_opt = np.array(sol['x'][N:]).reshape(-1)

    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez(f"baseline_solutions/linear.npz", x=x_opt, u=u_opt, t=t)

    return float(sol['f']), toc - tic

def solve_oscillatory():
    x = ca.MX.sym('x', N)
    u = ca.MX.sym('u', N)
    f_cos = np.cos(4*np.pi*t)
    cost = 0.5 * dt * ca.sumsqr(x) + 0.5 * dt * ca.sumsqr(u)
    constraints = [x[0]]
    for i in range(N - 1):
        dx = f_cos[i] + u[i]
        constraints.append(x[i+1] - (x[i] + dt * dx))
    constraints.append(x[-1])  # terminal condition
    g = ca.vertcat(*constraints)
    vars = ca.vertcat(x, u)
    prob = {'x': vars, 'f': cost, 'g': g}
    tic = time.time()
    solver = ca.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level': 0, 'print_time': 0})
    sol = solver(x0=np.zeros(2*N), lbg=0, ubg=0)
    toc = time.time()
    x_opt = np.array(sol['x'][:N]).reshape(-1)
    u_opt = np.array(sol['x'][N:]).reshape(-1)

    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez(f"baseline_solutions/oscillatory.npz", x=x_opt, u=u_opt, t=t)

    return float(sol['f']), toc - tic

def solve_polynomial_tracking():
    x = ca.MX.sym('x', N)
    u = ca.MX.sym('u', N)
    cost = dt * ca.sumsqr(x - t**2) + dt * ca.sumsqr(u)
    constraints = [x[0]]
    for i in range(N - 1):
        dx = u[i]
        constraints.append(x[i+1] - (x[i] + dt * dx))
    g = ca.vertcat(*constraints)
    vars = ca.vertcat(x, u)
    prob = {'x': vars, 'f': cost, 'g': g}
    tic = time.time()
    solver = ca.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level': 0, 'print_time': 0})
    sol = solver(x0=np.zeros(2*N), lbg=0, ubg=0)
    toc = time.time()

    x_opt = np.array(sol['x'][:N]).reshape(-1)
    u_opt = np.array(sol['x'][N:]).reshape(-1)

    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez(f"baseline_solutions/polynomial_tracking.npz", x=x_opt, u=u_opt, t=t)

    return float(sol['f']), toc - tic

def solve_nonlinear():
    x = ca.MX.sym('x', N)
    u = ca.MX.sym('u', N)
    cost = -x[-1]
    constraints = [x[0] - 1]
    for i in range(N - 1):
        dx = 2.5 * (-x[i] + x[i]*u[i] - u[i]**2)
        constraints.append(x[i+1] - (x[i] + dt * dx))
    g = ca.vertcat(*constraints)
    vars = ca.vertcat(x, u)
    prob = {'x': vars, 'f': cost, 'g': g}   
    tic = time.time()
    solver = ca.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level': 0, 'print_time': 0})
    sol = solver(x0=np.zeros(2*N), lbg=0, ubg=0)
    toc = time.time()
    x_opt = np.array(sol['x'][:N]).reshape(-1)
    u_opt = np.array(sol['x'][N:]).reshape(-1)

    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez(f"baseline_solutions/nonlinear.npz", x=x_opt, u=u_opt, t=t)

    return float(sol['f']), toc - tic

def solve_singular_arc():
    x = ca.MX.sym('x', N)
    u = ca.MX.sym('u', N)
    cost = dt * ca.sumsqr(u)
    constraints = [x[0] - 1]
    for i in range(N - 1):
        dx = x[i]**2 + u[i]
        constraints.append(x[i+1] - (x[i] + dt * dx))
    constraints.append(x[-1])  # terminal condition
    g = ca.vertcat(*constraints)
    vars = ca.vertcat(x, u)
    prob = {'x': vars, 'f': cost, 'g': g}
    tic = time.time()
    solver = ca.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level': 0, 'print_time': 0})
    sol = solver(x0=np.zeros(2*N), lbg=0, ubg=0)
    toc = time.time()
    x_opt = np.array(sol['x'][:N]).reshape(-1)
    u_opt = np.array(sol['x'][N:]).reshape(-1)

    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez(f"baseline_solutions/singular_arc.npz", x=x_opt, u=u_opt, t=t)

    return float(sol['f']), toc - tic

if __name__ == "__main__":
    if __name__ == "__main__":
        print("Benchmark     | Cost        | Time (s)")
        for name, func in [
            ("Linear", solve_linear),
            ("Oscillatory", solve_oscillatory),
            ("Polynomial", solve_polynomial_tracking),
            ("Nonlinear", solve_nonlinear),
            ("SingularArc", solve_singular_arc),
        ]:
            J, t_solve = func()
            print(f"{name:13s} | {J:10.6f} | {t_solve:.4f}")