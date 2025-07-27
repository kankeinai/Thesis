import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from models.fno import FNO1d
import os
import time


@dataclass
class PDEControlConfig:
    """Configuration for PDE control problems"""
    # Model parameters
    model_path: str
    modes: int = 32
    width: int = 64
    depth: int = 5
    in_dim: int = 2
    out_dim: int = 100
    activation: str = 'gelu'
    
    # Optimization parameters
    num_epochs: int = 30000
    learning_rate: float = 1e-3
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    initial_weight: float = 1.0
    objective_weight: float = 1
    rho: float = 1e-3
    # Early stopping parameters
    early_stopping: bool = True
    patience: int = 500
    min_delta: float = 1e-6
    monitor: str = 'total_loss'  # 'total_loss', 'obj', 'physics_loss'
    
    # Physical parameters
    nu: float = 0.01  # viscosity/diffusion coefficient
    alpha: float = 0.01  # reaction coefficient (for diffusion-reaction)
    
    # Grid parameters
    nx: int = 64
    nt: int = 100
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


class PDEControlSolver:
    """Base class for PDE control problems"""
    
    def __init__(self, config: PDEControlConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup grid
        self.x_np = np.linspace(config.x_min, config.x_max, config.nx)
        self.t_np = np.linspace(config.t_min, config.t_max, config.nt)
        self.dx = self.x_np[1] - self.x_np[0]
        self.dt = self.t_np[1] - self.t_np[0]
        
        # Setup tensors
        self.x = torch.tensor(self.x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Analytics storage
        self.analytics = {
            'obj': [],
            'total_loss': [],
            'physics_loss': [],
            'boundary_loss': [],
            'initial_loss': [],
            'control_loss': []
        }
    
    def _load_model(self) -> FNO1d:
        """Load the trained FNO model"""
        model = FNO1d(
            modes=self.config.modes,
            width=self.config.width,
            depth=self.config.depth,
            in_dim=self.config.in_dim,
            out_dim=self.config.out_dim,
            activation=self.config.activation
        ).to(self.device)
        
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        return model
    
    def compute_residual(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual - to be implemented by subclasses"""
        raise NotImplementedError
    
    def compute_initial_loss(self, y: torch.Tensor) -> torch.Tensor:
        """Compute initial condition loss"""
        return torch.mean(y[:, :, 0]**2)
    
    def compute_boundary_loss(self, y: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss"""
        return torch.mean(y[:, 0, :]**2) + torch.mean(y[:, -1, :]**2)
    
    def compute_control_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Compute control regularization loss"""
        return torch.mean(u**2)* self.config.rho
    
    def compute_objective(self, y: torch.Tensor, u: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute objective function"""
        final_y = y[:, :, -1]
        tracking_loss = torch.mean((final_y - target.unsqueeze(0))**2)
        control_loss = self.compute_control_loss(u)
        return tracking_loss/2*self.config.objective_weight +  control_loss/2
    
    def solve(self, target_profile: torch.Tensor, initial_guess: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Solve the control problem"""
        if initial_guess is None:
            initial_guess = torch.rand((1, self.config.nx), dtype=torch.float32, device=self.device)
        
        u_param = torch.nn.Parameter(initial_guess)
        optimizer = optim.Adam([u_param], lr=self.config.learning_rate)
        
        print(f"Starting optimization for {self.config.num_epochs} epochs...")
        
        # Start timing
        start_time = time.time()
        
        # Early stopping variables
        best_loss = float('inf')
        wait = 0
        best_u = None
        best_y = None
        
        for epoch in range(1, self.config.num_epochs + 1):
            optimizer.zero_grad()
            
            # Forward pass
            y = self.model(u_param, self.x)
            
            # Compute losses
            obj = self.compute_objective(y, u_param, target_profile)
            residual = self.compute_residual(y, u_param)
            physics_loss = torch.mean(residual**2)
            initial_loss = self.compute_initial_loss(y)
            boundary_loss = self.compute_boundary_loss(y)
            
            # Total loss
            loss = (self.config.physics_weight * physics_loss +
                   self.config.initial_weight * initial_loss +
                   self.config.boundary_weight * boundary_loss +
                obj)
            
            # Store analytics
            self.analytics['obj'].append(obj.item())
            self.analytics['total_loss'].append(loss.item())
            self.analytics['physics_loss'].append(physics_loss.item())
            self.analytics['boundary_loss'].append(boundary_loss.item())
            self.analytics['initial_loss'].append(initial_loss.item())
            self.analytics['control_loss'].append(self.compute_control_loss(u_param).item())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Early stopping logic
            if self.config.early_stopping:
                current_loss = self.analytics[self.config.monitor][-1]
                
                if current_loss < best_loss - self.config.min_delta:
                    best_loss = current_loss
                    wait = 0
                    best_u = u_param.detach().clone()
                    best_y = y.detach().clone()
                else:
                    wait += 1
                
                if wait >= self.config.patience:
                    print(f"Early stopping at epoch {epoch} (patience: {self.config.patience})")
                    print(f"Best {self.config.monitor}: {best_loss:.6f}")
                    return {
                        'u_optimal': best_u,
                        'y_optimal': best_y,
                        'analytics': self.analytics,
                        'stopped_early': True,
                        'best_epoch': epoch - self.config.patience
                    }
            
            # Logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Physics: {physics_loss.item():.6f}")
                if self.config.early_stopping:
                    print(f"  Best {self.config.monitor}: {best_loss:.6f} | Wait: {wait}")
        
        # End timing
        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        
        return {
            'u_optimal': u_param.detach(),
            'y_optimal': y.detach(),
            'analytics': self.analytics,
            'stopped_early': False,
            'best_epoch': self.config.num_epochs,
            'optimization_time': optimization_time
        }
    
    def plot_results(self, results: Dict[str, Any], target_profile: torch.Tensor, save_path: Optional[str] = None):
        """Plot optimization results"""
        u_opt = results['u_optimal'].cpu().numpy()[0]
        y_opt = results['y_optimal'].cpu().numpy()[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Optimal control
        axes[0, 0].plot(self.x_np, u_opt, 'b-', linewidth=2, label='Optimal Control')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('u(x)')
        axes[0, 0].set_title('Optimal Control')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Final state vs target
        axes[0, 1].plot(self.x_np, y_opt[:, -1], 'r-', linewidth=2, label='Final State')
        axes[0, 1].plot(self.x_np, target_profile.cpu().numpy(), 'g--', linewidth=2, label='Target')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y(x, T)')
        axes[0, 1].set_title('Final State vs Target')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # State evolution
        im = axes[1, 0].imshow(y_opt, aspect='auto', origin='lower', 
                              extent=[self.config.t_min, self.config.t_max, self.config.x_min, self.config.x_max])
        axes[1, 0].set_xlabel('t')
        axes[1, 0].set_ylabel('x')
        axes[1, 0].set_title('State Evolution')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Loss history
        axes[1, 1].plot(self.analytics['total_loss'], 'b-', label='Total Loss')
        axes[1, 1].plot(self.analytics['obj'], 'r-', label='Objective')
        axes[1, 1].plot(self.analytics['physics_loss'], 'g-', label='Physics Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss History')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Save solutions in the same directory as the plot
            self.save_solutions(results, target_profile, save_path)
        plt.show()

    def save_solutions(self, results: Dict[str, Any], target_profile: torch.Tensor, plot_path: str):
        """Save the found solutions (u and y) to a folder"""
        import os
        
        # Create solutions directory based on plot path
        plot_dir = os.path.dirname(plot_path)
        plot_name = os.path.splitext(os.path.basename(plot_path))[0]
        solutions_dir = os.path.join(plot_dir, f"{plot_name}_solutions")
        os.makedirs(solutions_dir, exist_ok=True)
        
        # Extract solutions
        u_opt = results['u_optimal'].cpu().numpy()
        y_opt = results['y_optimal'].cpu().numpy()
        target = target_profile.cpu().numpy()
        
        # Save solutions as numpy files
        np.save(os.path.join(solutions_dir, 'u_optimal.npy'), u_opt)
        np.save(os.path.join(solutions_dir, 'y_optimal.npy'), y_opt)
        np.save(os.path.join(solutions_dir, 'target_profile.npy'), target)
        
        # Save grid information
        grid_info = {
            'x_grid': self.x_np,
            't_grid': self.t_np,
            'dx': self.dx,
            'dt': self.dt
        }
        np.save(os.path.join(solutions_dir, 'grid_info.npy'), grid_info)
        
        # Save optimization analytics
        np.save(os.path.join(solutions_dir, 'analytics.npy'), self.analytics)
        
        # Save configuration
        config_dict = {
            'model_path': self.config.model_path,
            'num_epochs': self.config.num_epochs,
            'learning_rate': self.config.learning_rate,
            'physics_weight': self.config.physics_weight,
            'boundary_weight': self.config.boundary_weight,
            'initial_weight': self.config.initial_weight,
            'objective_weight': self.config.objective_weight,
            'rho': self.config.rho,
            'nu': self.config.nu,
            'alpha': self.config.alpha,
            'nx': self.config.nx,
            'nt': self.config.nt
        }
        np.save(os.path.join(solutions_dir, 'config.npy'), config_dict)
        
        # Create a summary text file
        summary_path = os.path.join(solutions_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"PDE Control Solution Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Model: {self.config.model_path}\n")
            f.write(f"Grid: {self.config.nx} x {self.config.nt} points\n")
            f.write(f"Domain: x ∈ [{self.config.x_min}, {self.config.x_max}], t ∈ [{self.config.t_min}, {self.config.t_max}]\n")
            f.write(f"Optimization epochs: {len(self.analytics['total_loss'])}\n")
            f.write(f"Final objective: {self.analytics['obj'][-1]:.6e}\n")
            f.write(f"Final physics loss: {self.analytics['physics_loss'][-1]:.6e}\n")
            if 'optimization_time' in results:
                f.write(f"Optimization time: {results['optimization_time']:.2f} seconds\n")
            if 'stopped_early' in results:
                f.write(f"Early stopping: {results['stopped_early']}\n")
                if results['stopped_early']:
                    f.write(f"Best epoch: {results['best_epoch']}\n")
            f.write(f"\nFiles saved:\n")
            f.write(f"- u_optimal.npy: Optimal control function\n")
            f.write(f"- y_optimal.npy: Optimal state evolution\n")
            f.write(f"- target_profile.npy: Target profile\n")
            f.write(f"- grid_info.npy: Grid information\n")
            f.write(f"- analytics.npy: Optimization history\n")
            f.write(f"- config.npy: Configuration parameters\n")
        
        print(f"Solutions saved to: {solutions_dir}")

    def animate_solution(self, results: Dict[str, Any], target_profile: torch.Tensor, 
                        save_path: Optional[str] = None, title: str = "PDE Control Solution"):
        """
        Create an animation showing the control solution and trajectory evolution.
        
        Args:
            results: Dictionary containing 'u_optimal' and 'y_optimal'
            target_profile: Target profile to reach
            save_path: Path to save the animation (optional)
            title: Title for the animation
        """
        u_optimal = results.get('u_optimal', results.get('f_optimal'))
        y_optimal = results.get('y_optimal', results.get('u_optimal'))
        
        if u_optimal is None or y_optimal is None:
            print("❌ No optimal control or state found in results")
            return
        
        # Convert to numpy
        u_np = u_optimal.cpu().numpy() if torch.is_tensor(u_optimal) else u_optimal
        y_np = y_optimal.cpu().numpy() if torch.is_tensor(y_optimal) else y_optimal
        target_np = target_profile.cpu().numpy() if torch.is_tensor(target_profile) else target_profile
        
        # Ensure correct shapes
        if len(u_np.shape) == 1:
            u_np = u_np.reshape(-1, 1)  # [Nx, 1]
        if len(y_np.shape) == 3 and y_np.shape[0] == 1:
            y_np = y_np.squeeze(0)  # Remove batch dimension
        if len(y_np.shape) == 2:
            y_np = y_np.reshape(y_np.shape[0], y_np.shape[1], 1)  # [Nx, Nt, 1]
        if len(target_np.shape) == 1:
            target_np = target_np.reshape(-1, 1)  # [Nx, 1]
        
        # Set up the figure
        fig, (ax_u, ax_y) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot control input (static)
        ax_u.plot(self.x_np, u_np.flatten(), 'b-', linewidth=2, label='Optimal Control')
        ax_u.set_xlabel('x')
        ax_u.set_ylabel('u(x)')
        ax_u.set_title('Optimal Control Input')
        ax_u.grid(True)
        ax_u.legend()
        
        # Set up animation for state evolution
        ax_y.set_xlim(self.config.x_min, self.config.x_max)
        ax_y.set_xlabel('x')
        ax_y.set_ylabel('y(x,t)')
        ax_y.set_title(f'{title} - State Evolution')
        ax_y.grid(True)
        
        # Plot target profile (static)
        ax_y.plot(self.x_np, target_np.flatten(), 'r--', linewidth=2, label='Target Profile')
        
        # Initialize lines for animation
        state_line, = ax_y.plot([], [], 'b-', linewidth=2, label='Current State')
        ax_y.legend()
        
        def init():
            state_line.set_data([], [])
            return state_line,
        
        def animate(t):
            # Get current state at time t
            if len(y_np.shape) == 3:
                current_state = y_np[:, t, 0] if y_np.shape[2] == 1 else y_np[:, t]
            else:
                current_state = y_np[:, t]
            
            state_line.set_data(self.x_np, current_state)
            
            # Update title with current time
            current_time = self.t_np[t]
            ax_y.set_title(f'{title} - State Evolution (t={current_time:.2f})')
            
            return state_line,
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.t_np), 
            init_func=init, blit=True, interval=100
        )
        
        if save_path:
            # Save as GIF
            anim.save(save_path, writer='pillow', fps=10)
            print(f"✅ Saved animation to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return anim


class HeatControlSolver(PDEControlSolver):
    """Solver for heat equation control: ∂y/∂t = ν ∂²y/∂x² + u(x)"""
    
    def compute_residual(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute heat equation residual"""
        # ∂y/∂t
        dudt = torch.gradient(y, spacing=self.dt, dim=2)[0]
        
        # ∂²y/∂x²
        dy_dx = torch.gradient(y, spacing=self.dx, dim=1)[0]
        d2y_dx2 = torch.gradient(dy_dx, spacing=self.dx, dim=1)[0]
        
        # Expand u to match [B, Nx, Nt]
        u_expanded = u.unsqueeze(-1).expand_as(y)
        
        return dudt - self.config.nu * d2y_dx2 - u_expanded


class DiffusionReactionControlSolver(PDEControlSolver):
    """Solver for diffusion-reaction equation control: ∂y/∂t = ν ∂²y/∂x² - α y² + u(x)"""
    
    def compute_residual(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute diffusion-reaction equation residual"""
        # ∂y/∂t
        dudt = torch.gradient(y, spacing=self.dt, dim=2)[0]
        
        # ∂²y/∂x²
        dy_dx = torch.gradient(y, spacing=self.dx, dim=1)[0]
        d2y_dx2 = torch.gradient(dy_dx, spacing=self.dx, dim=1)[0]
        
        # Nonlinear reaction term
        y_sq = y**2
        
        # Expand u to match [B, Nx, Nt]
        u_expanded = u.unsqueeze(-1).expand_as(y)
        
        return dudt - self.config.nu * d2y_dx2 + self.config.alpha * y_sq - u_expanded


class BurgersControlSolver(PDEControlSolver):
    """Solver for Burgers equation control: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f(x)"""
    
    def compute_residual(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute Burgers equation residual"""
        # ∂y/∂t
        dydt = torch.gradient(y, spacing=self.dt, dim=2)[0]
        
        # ∂y/∂x and ∂²y/∂x²
        dydx = torch.gradient(y, spacing=self.dx, dim=1)[0]
        d2ydx2 = torch.gradient(dydx, spacing=self.dx, dim=1)[0]
        
        # Expand u to match [B, Nx, Nt]
        u_expanded = u.unsqueeze(-1).expand_as(y)
        
        return dydt + y * dydx - self.config.nu * d2ydx2 - u_expanded
    
    def solve(self, target_profile: torch.Tensor, initial_guess: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Override solve method for Burgers equation"""
        if initial_guess is None:
            initial_guess = torch.rand((1, self.config.nx), dtype=torch.float32, device=self.device) * 1.5
        
        u_param = torch.nn.Parameter(initial_guess)  # Control variable u(x)
        optimizer = optim.Adam([u_param], lr=self.config.learning_rate)
        
        print(f"Starting Burgers optimization for {self.config.num_epochs} epochs...")
        
        # Start timing
        start_time = time.time()
        
        # Early stopping variables
        best_loss = float('inf')
        wait = 0
        best_u = None
        best_y = None
        
        for epoch in range(1, self.config.num_epochs + 1):
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(u_param, self.x)  # State variable y(x,t)
            
            # Compute losses
            tracking_loss = 0.5 * torch.mean((y_pred[:, :, -1] - target_profile.unsqueeze(0))**2) * self.config.objective_weight
            residual = self.compute_residual(y_pred, u_param)
            physics_loss = torch.mean(residual**2)
            initial_loss = self.compute_initial_loss(y_pred)
            boundary_loss = self.compute_boundary_loss(y_pred)
            control_loss = torch.mean(u_param**2) * self.config.rho
            
            # Total loss
            loss = (tracking_loss + 
                   self.config.physics_weight * physics_loss +
                   self.config.initial_weight * initial_loss +
                   self.config.boundary_weight * boundary_loss +
                   control_loss)
            
            # Store analytics
            self.analytics['obj'].append(tracking_loss.item())
            self.analytics['total_loss'].append(loss.item())
            self.analytics['physics_loss'].append(physics_loss.item())
            self.analytics['boundary_loss'].append(boundary_loss.item())
            self.analytics['initial_loss'].append(initial_loss.item())
            self.analytics['control_loss'].append(control_loss.item())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Early stopping logic
            if self.config.early_stopping:
                current_loss = self.analytics[self.config.monitor][-1]
                
                if current_loss < best_loss - self.config.min_delta:
                    best_loss = current_loss
                    wait = 0
                    best_u = u_param.detach().clone()
                    best_y = y_pred.detach().clone()
                else:
                    wait += 1
                
                if wait >= self.config.patience:
                    print(f"Early stopping at epoch {epoch} (patience: {self.config.patience})")
                    print(f"Best {self.config.monitor}: {best_loss:.6f}")
                    return {
                        'u_optimal': best_u,
                        'y_optimal': best_y,
                        'analytics': self.analytics,
                        'stopped_early': True,
                        'best_epoch': epoch - self.config.patience
                    }
            
            # Logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Physics: {physics_loss.item():.6f}")
                if self.config.early_stopping:
                    print(f"  Best {self.config.monitor}: {best_loss:.6f} | Wait: {wait}")
        
        # End timing
        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        
        return {
            'u_optimal': u_param.detach(),
            'y_optimal': y_pred.detach(),
            'analytics': self.analytics,
            'stopped_early': False,
            'best_epoch': self.config.num_epochs,
            'optimization_time': optimization_time
        }


def create_target_profiles():
    """Create different target profiles for testing"""
    
    def gaussian_target(x, A=0.5, x0=0.5, sigma=0.1):
        return torch.tensor(
            A * np.exp(-((x - x0)**2) / (2 * sigma**2)),
            dtype=torch.float32
        )
    
    def double_gaussian_target(x, A=1.0, sigma=0.05):
        x = torch.tensor(x, dtype=torch.float32)
        bump1 = A * torch.exp(-((x - 0.3)**2) / (2 * sigma**2))
        bump2 = A * torch.exp(-((x - 0.7)**2) / (2 * sigma**2))
        return bump1 + bump2
    
    def shock_target(x, location=0.5, steepness=50):
        x = torch.tensor(x, dtype=torch.float32)
        return (1.0 / (1.0 + torch.exp(-steepness * (x - location)))) * 0.1
    
    return {
        'gaussian': gaussian_target,
        'double_gaussian': double_gaussian_target,
        'shock': shock_target
    }


def main():
    """Example usage of the structured PDE control solver"""
    
    # Create results directory
    results_dir = "control_pde_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create target profiles
    target_profiles = create_target_profiles()
    x_np = np.linspace(0, 1, 64)
    
    # Example 1: Heat equation control
    print("=== Heat Equation Control ===")
    heat_config = PDEControlConfig(
        model_path="trained_models/heat1d/ckpt_ep0100.pt",
        num_epochs=20000,  # Reduced for demo
        physics_weight=1,
        boundary_weight=0.01,
        initial_weight=0.01,
        objective_weight=10,
        learning_rate=1e-3,
        depth=4  # Heat model was trained with depth=4
    )
    
    heat_solver = HeatControlSolver(heat_config)
    target = target_profiles['gaussian'](x_np)
    results = heat_solver.solve(target)
    opt_time = results.get('optimization_time')
    if opt_time is not None:
        print(f"Heat equation optimization completed in {opt_time:.2f} seconds")
    else:
        print("Heat equation optimization completed (timing not available)")
    heat_solver.plot_results(results, target, os.path.join(results_dir, "heat_control_results.png"))
    
    # Create animation for heat equation
    print("Creating heat equation animation...")
    heat_solver.animate_solution(results, target, os.path.join(results_dir, "heat_control_animation.gif"), "Heat Equation Control")
    
    # Example 2: Diffusion-reaction control
    print("\n=== Diffusion-Reaction Control ===")
    diff_config = PDEControlConfig(
        model_path="trained_models/diffusion1d/ckpt_ep0300.pt",
        num_epochs=20000,  # Reduced for demo
        learning_rate=1e-3,
        depth=5,  # Diffusion model was trained with depth=5
        physics_weight=1,
        boundary_weight=0.01,
        initial_weight=0.01,
        objective_weight=1
    )
    
    diff_solver = DiffusionReactionControlSolver(diff_config)
    target = target_profiles['double_gaussian'](x_np)
    results = diff_solver.solve(target)
    opt_time = results.get('optimization_time')
    if opt_time is not None:
        print(f"Diffusion-reaction optimization completed in {opt_time:.2f} seconds")
    else:
        print("Diffusion-reaction optimization completed (timing not available)")
    diff_solver.plot_results(results, target, os.path.join(results_dir, "diffusion_control_results.png"))
    
    # Create animation for diffusion-reaction equation
    print("Creating diffusion-reaction animation...")
    diff_solver.animate_solution(results, target, os.path.join(results_dir, "diffusion_control_animation.gif"), "Diffusion-Reaction Control")
    
    # Example 3: Burgers equation control
    print("\n=== Burgers Equation Control ===")
    burgers_config = PDEControlConfig(
        model_path="trained_models/burgers1d/best.pt",
        num_epochs=20000,  # Reduced for demo
        learning_rate=1e-3,
        depth=5,  # Burgers model was trained with depth=5
        physics_weight=5,
        boundary_weight=1,
        initial_weight=1,
        objective_weight=10,
    )
    
    burgers_solver = BurgersControlSolver(burgers_config)
    target = target_profiles['shock'](x_np)
    results = burgers_solver.solve(target)
    opt_time = results.get('optimization_time')
    if opt_time is not None:
        print(f"Burgers equation optimization completed in {opt_time:.2f} seconds")
    else:
        print("Burgers equation optimization completed (timing not available)")
    burgers_solver.plot_results(results, target, os.path.join(results_dir, "burgers_control_results.png"))
    
    # Create animation for Burgers equation
    print("Creating Burgers equation animation...")
    burgers_solver.animate_solution(results, target, os.path.join(results_dir, "burgers_control_animation.gif"), "Burgers Equation Control")


if __name__ == "__main__":
    main() 