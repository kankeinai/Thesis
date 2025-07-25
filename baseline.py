import casadi as ca
import numpy as np
import time 
from pathlib import Path
from utils.settings import optimal_solutions

N = 200
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
    true_x_opt = optimal_solutions['linear']['x']({'t':t})
    true_u_opt = optimal_solutions['linear']['u']({'t':t})

    relative_error_x = np.linalg.norm(x_opt - true_x_opt) / np.linalg.norm(true_x_opt)
    relative_error_u = np.linalg.norm(u_opt - true_u_opt) / np.linalg.norm(true_u_opt)
    print("Problem: linear")
    print(f"Relative error in x: {relative_error_x},Relative error in u: {relative_error_u}")

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

    true_x_opt = optimal_solutions['oscillatory']['x']({'t':t})
    true_u_opt = optimal_solutions['oscillatory']['u']({'t':t})

    relative_error_x = np.linalg.norm(x_opt - true_x_opt) / np.linalg.norm(true_x_opt)
    relative_error_u = np.linalg.norm(u_opt - true_u_opt) / np.linalg.norm(true_u_opt)
    print("Problem: oscillatory")
    print(f"Relative error in x: {relative_error_x},Relative error in u: {relative_error_u}")

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

    true_x_opt = optimal_solutions['polynomial_tracking']['x']({'t':t})
    true_u_opt = optimal_solutions['polynomial_tracking']['u']({'t':t})

    relative_error_x = np.linalg.norm(x_opt - true_x_opt) / np.linalg.norm(true_x_opt)
    relative_error_u = np.linalg.norm(u_opt - true_u_opt) / np.linalg.norm(true_u_opt)
    print("Problem: polynomial_tracking")
    print(f"Relative error in x: {relative_error_x},Relative error in u: {relative_error_u}")

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

    true_x_opt = optimal_solutions['nonlinear']['x']({'t':t})
    true_u_opt = optimal_solutions['nonlinear']['u']({'t':t})

    relative_error_x = np.linalg.norm(x_opt - true_x_opt) / np.linalg.norm(true_x_opt)
    relative_error_u = np.linalg.norm(u_opt - true_u_opt) / np.linalg.norm(true_u_opt)
    print("Problem: nonlinear")
    print(f"Relative error in x: {relative_error_x},Relative error in u: {relative_error_u}")



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

    true_x_opt = optimal_solutions['singular_arc']['x']({'t':t})
    true_u_opt = optimal_solutions['singular_arc']['u']({'t':t})

    relative_error_x = np.linalg.norm(x_opt - true_x_opt) / np.linalg.norm(true_x_opt)
    relative_error_u = np.linalg.norm(u_opt - true_u_opt) / np.linalg.norm(true_u_opt)
    print("Problem: singular_arc")
    print(f"Relative error in x: {relative_error_x},Relative error in u: {relative_error_u}")

    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez(f"baseline_solutions/singular_arc.npz", x=x_opt, u=u_opt, t=t)

    return float(sol['f']), toc - tic

# ---------------------------------------------------------------------
# Burgers distributed–forcing optimal control
# grid:  nx = 64 (space)   nt = 200 (time)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Burgers distributed‑forcing optimal control  (nx=64, nt=200)
# ---------------------------------------------------------------------
def solve_burgers_control():
    """
    Minimise a quadratic tracking + control‑effort cost subject to the
    1‑D viscous Burgers PDE:
        u_t + u u_x = ν u_xx + f
    Boundary: u(0,t)=u(1,t)=0,  Initial: u(x,0)=0
    Target profile at T=1:  u_d(x) = sin(π x)
    """
    # Discretisation ---------------------------------------------------
    Nx, Nt = 64, 200
    Lx, T_final = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dt = T_final / (Nt - 1)
    x = np.linspace(0, Lx, Nx)
    t = np.linspace(0, T_final, Nt)

    ν      = 0.02        # viscosity inside training range
    λ_ctrl = 1e-3        # control‑effort weight

    u_des = np.sin(np.pi * x)     # desired terminal profile

    # CasADi variables -------------------------------------------------
    U = ca.MX.sym("U", Nt, Nx)        # state  (Nt × Nx)
    F = ca.MX.sym("F", Nt-1, Nx)      # control (Nt‑1 × Nx), piecewise‑constant

    # Objective --------------------------------------------------------
    J_track = 0.5 * ca.sumsqr(U[-1, :] - u_des)
    J_reg   = 0.5 * λ_ctrl * dt * ca.sumsqr(F)
    J       = J_track + J_reg

    # Finite‑difference helpers ---------------------------------------
    def laplacian_1d(u_row):
        """Second spatial derivative with Dirichlet BCs (ghost nodes = 0)."""
        return ca.hcat([
            0,
            u_row[2:] - 2*u_row[1:-1] + u_row[:-2],
            0
        ]) / dx**2

    def advection_1d(u_row):
        """Central‑difference u * u_x with first/last node one‑sided."""
        dudx = ca.hcat([
            (u_row[1]   - u_row[0])   / dx,
            (u_row[2:]  - u_row[:-2]) / (2*dx),
            (u_row[-1]  - u_row[-2])  / dx
        ])
        return u_row * dudx

    # Dynamics constraints --------------------------------------------
    g = [U[0, :]]                       # enforce u(x,0)=0
    for k in range(Nt - 1):
        rhs = -advection_1d(U[k, :]) + ν * laplacian_1d(U[k, :]) + F[k, :]
        g.append(U[k+1, :] - (U[k, :] + dt * rhs))
    g = ca.vertcat(*[ca.reshape(row, -1, 1) for row in g])

    # NLP --------------------------------------------------------------
    Z     = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(F, -1, 1))
    prob  = {'x': Z, 'f': J, 'g': g}
    opts  = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    x0   = np.zeros(Z.shape[0])         # cold start
    tic  = time.time()
    sol  = solver(x0=x0, lbg=0, ubg=0)
    toc  = time.time()

    z_opt = np.array(sol['x']).flatten()
    U_opt = z_opt[:Nt * Nx].reshape(Nt, Nx)
    F_opt = z_opt[Nt * Nx:].reshape(Nt-1, Nx)

    # -----------------------------------------------------------------
    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez("baseline_solutions/burgers.npz",
             u=U_opt, f=F_opt, x=x, t=t)

    print("Problem: Burgers (ν = 0.02)")
    print(f"Optimal cost: {float(sol['f']):.6f}, Solve time: {toc - tic:.2f}s")

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
            ("Burgers", solve_burgers_control),  
        ]:
            J, t_solve = func()
            print(f"{name:13s} | {J:10.6f} | {t_solve:.4f}")