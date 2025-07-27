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

def solve_heat_control():
    """
    Minimize quadratic tracking + control effort cost subject to 1D heat PDE:
        y_t = nu y_xx + u(x,t)
    Boundary: y(0,t) = y(1,t) = 0,
    Initial: y(x,0) = 0,
    Target at T=1: Gaussian bump,
    Control: time- and space-dependent.
    """

    # Discretization
    Nx, Nt = 64, 100
    Lx, T_final = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dt = T_final / (Nt - 1)
    x = np.linspace(0, Lx, Nx)
    t = np.linspace(0, T_final, Nt)

    nu = 0.01          # diffusivity
    rho = 1e-3         # control regularization weight

    # Desired terminal state: Gaussian bump
    def y_desired(x, A=1.0, x0=0.5, sigma=0.1):
        return A * np.exp(-((x - x0)**2) / (2 * sigma**2))

    y_des = y_desired(x)
    y_des_casadi = ca.DM(y_des).T    # 1 x Nx row vector

    # CasADi variables
    Y = ca.MX.sym("Y", Nt, Nx)       # state (Nt x Nx)
    U = ca.MX.sym("U", Nx)      # control (Nt x Nx)

    # Objective: tracking at final time + control effort over all time and space
    J_track = 0.5 * ca.sumsqr(Y[-1, :] - y_des_casadi)
    J_reg = 0.5 * rho * T_final * ca.sumsqr(U)
    J = J_track + J_reg

    # Laplacian with Dirichlet BCs (ghost nodes = 0)
    def laplacian_1d(y_row):
        return ca.hcat([
            0,
            y_row[2:] - 2 * y_row[1:-1] + y_row[:-2],
            0
        ]) / dx**2

    # PDE constraints using implicit Euler:
    # Y[k+1] - Y[k] = dt * (nu * Y_xx[k+1] + U[k])
    g = [Y[0, :]]  # initial condition y(x,0)=0

    for k in range(Nt - 1):
        lap = laplacian_1d(Y[k + 1, :])
        lap_col = ca.reshape(lap, Nx, 1)
        rhs = nu * lap_col + ca.reshape(U, Nx, 1)   # time-invariant U
        rhs_row = rhs.T
        g.append(Y[k + 1, :] - (Y[k, :] + dt * rhs_row))

    g = ca.vertcat(*[ca.reshape(row, -1, 1) for row in g])

    # NLP setup
    Z = ca.vertcat(ca.reshape(Y, -1, 1), U)   # (Nt*Nx*2) x 1
    prob = {'x': Z, 'f': J, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    # Initial guess
    x0 = np.zeros(Z.shape[0])

    tic = time.time()
    sol = solver(x0=x0, lbg=0, ubg=0)
    toc = time.time()

    z_opt = np.array(sol['x']).flatten()
    Y_opt = z_opt[:Nt * Nx].reshape(Nt, Nx)
    U_opt = z_opt[Nt * Nx:] 

    # Save solution
    np.savez("baseline_solutions/heat_control_time_invariant.npz",
         y=Y_opt, u=U_opt, x=x, t=t)

    print("Problem: Heat equation with time-invariant control (ν = 0.01)")
    print(f"Optimal cost: {float(sol['f']):.6f}, Solve time: {toc - tic:.2f}s")

    return float(sol['f']), toc - tic

def solve_diffusion_control():
    """
    Solve optimal control for the 1D diffusion-reaction equation:
        ∂y/∂t = ν ∂²y/∂x² - α y² + u(x)
    with initial condition y(x,0)=0 and Dirichlet BCs y(0,t)=y(1,t)=0.
    """

    # Discretization
    Nx, Nt = 64, 100
    Lx, T_final = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dt = T_final / (Nt - 1)
    x = np.linspace(0, Lx, Nx)
    t = np.linspace(0, T_final, Nt)

    # Parameters
    nu = 0.01      # diffusivity
    alpha = 0.01   # reaction coefficient
    rho = 1e-3     # control regularization

    # Target profile: two Gaussian bumps
    def y_desired(x, A=1.0, sigma=0.05):
        bump1 = A * np.exp(-((x - 0.3)**2) / (2 * sigma**2))
        bump2 = A * np.exp(-((x - 0.7)**2) / (2 * sigma**2))
        return bump1 + bump2

    y_des = y_desired(x)
    y_des_casadi = ca.DM(y_des).T  # row vector shape (1, Nx)

    # CasADi variables
    Y = ca.MX.sym("Y", Nt, Nx)  # state trajectory
    U = ca.MX.sym("U", Nx)      # control (time-invariant)

    # Objective function
    J_track = 0.5 * ca.sumsqr(Y[-1, :] - y_des_casadi)
    J_reg = 0.5 * rho * T_final * ca.sumsqr(U)
    J = J_track + J_reg

    # Laplacian with Dirichlet BCs (zero padding)
    def laplacian_1d(y_row):
        return ca.hcat([
            0,
            y_row[2:] - 2 * y_row[1:-1] + y_row[:-2],
            0
        ]) / dx**2

    # PDE constraints
    g = [Y[0, :]]  # initial condition y(x,0) = 0

    for k in range(Nt - 1):
        y_next = Y[k + 1, :]
        lap = laplacian_1d(y_next)

        # Fix shape mismatch: ensure U is row vector
        rhs = nu * lap - alpha * (y_next ** 2) + U.T

        g.append(Y[k + 1, :] - (Y[k, :] + dt * rhs))

    # Flatten constraint list
    g = ca.vertcat(*[ca.reshape(row, -1, 1) for row in g])

    # Optimization setup
    Z = ca.vertcat(ca.reshape(Y, -1, 1), U)
    prob = {'x': Z, 'f': J, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    # Initial guess
    x0 = np.zeros(Z.shape[0])

    # Solve
    tic = time.time()
    sol = solver(x0=x0, lbg=0, ubg=0)
    toc = time.time()

    # Extract solution
    z_opt = np.array(sol['x']).flatten()
    Y_opt = z_opt[:Nt * Nx].reshape(Nt, Nx)
    U_opt = z_opt[Nt * Nx:]

    # Save
    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez("baseline_solutions/diffusion_control.npz", y=Y_opt, u=U_opt, x=x, t=t)

    print("Problem: Diffusion-reaction with time-invariant control")
    print(f"Optimal cost: {float(sol['f']):.6f}, Solve time: {toc - tic:.2f}s")

    return float(sol['f']), toc - tic

def solve_burgers_control():
    """
    Solve optimal control for 1D viscous Burgers’ equation:
        ∂y/∂t + y ∂y/∂x = ν ∂²y/∂x² + u(x)
    with y(x,0)=0 and Dirichlet BCs y(0,t)=y(1,t)=0 (implicitly enforced).
    """

    # Discretization
    Nx, Nt = 64, 100
    Lx, T_final = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dt = T_final / (Nt - 1)
    x = np.linspace(0, Lx, Nx)
    t = np.linspace(0, T_final, Nt)

    # Parameters
    nu = 0.01    # viscosity
    rho = 1e-3   # control regularization

    # Target profile: shock-type sigmoid
    def y_desired(x, location=0.5, steepness=50, amplitude=0.1):
        return amplitude / (1.0 + np.exp(-steepness * (x - location)))

    y_des = y_desired(x)
    y_des_casadi = ca.DM(y_des).T  # shape (1, Nx)

    # CasADi decision variables
    Y = ca.MX.sym("Y", Nt, Nx)  # state trajectory
    U = ca.MX.sym("U", Nx)      # control (time-invariant)

    # Cost functional
    J_track = 0.5 * ca.sumsqr(Y[-1, :] - y_des_casadi)
    J_reg = 0.5 * rho * T_final * ca.sumsqr(U)
    J = J_track + J_reg

    # Central difference (gradient)
    def grad_1d(y_row):
        return ca.hcat([
            0,
            (y_row[2:] - y_row[:-2]) / (2 * dx),
            0
        ])

    # Laplacian with zero-padding (implicit Dirichlet BCs)
    def laplacian_1d(y_row):
        return ca.hcat([
            0,
            y_row[2:] - 2 * y_row[1:-1] + y_row[:-2],
            0
        ]) / dx**2

    # PDE constraints
    g = [Y[0, :]]  # initial condition: y(x,0) = 0

    for k in range(Nt - 1):
        y_next = Y[k + 1, :]
        dudx = grad_1d(y_next)
        lap = laplacian_1d(y_next)

        rhs = -y_next * dudx + nu * lap + U.T  # ensure shape (1, Nx)
        g.append(Y[k + 1, :] - (Y[k, :] + dt * rhs))

    g = ca.vertcat(*[ca.reshape(row, -1, 1) for row in g])

    # NLP definition
    Z = ca.vertcat(ca.reshape(Y, -1, 1), U)  # flatten Y then append U
    prob = {'x': Z, 'f': J, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    # Initial guess
    x0 = np.zeros(Z.shape[0])

    # Solve
    tic = time.time()
    sol = solver(x0=x0, lbg=0, ubg=0)
    toc = time.time()

    # Extract solution
    z_opt = np.array(sol['x']).flatten()
    Y_opt = z_opt[:Nt * Nx].reshape(Nt, Nx)
    U_opt = z_opt[Nt * Nx:]

    # Save
    Path("baseline_solutions").mkdir(exist_ok=True)
    np.savez("baseline_solutions/burgers_control.npz", y=Y_opt, u=U_opt, x=x, t=t)

    print("Problem: Viscous Burgers’ equation with time-invariant control")
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
            ("Heat", solve_heat_control),  
            ("Burgers", solve_burgers_control),
            ("Diffusion", solve_diffusion_control),
        ]:
            J, t_solve = func()
            print(f"{name:13s} | {J:10.6f} | {t_solve:.4f}")