#!/usr/bin/env python3
"""
Example script demonstrating the structured PDE control solver.
This shows how to solve control problems for different PDEs in a clean, modular way.
"""

import torch
import numpy as np
from pde_control_solver import (
    PDEControlConfig, 
    HeatControlSolver, 
    DiffusionReactionControlSolver, 
    BurgersControlSolver,
    create_target_profiles
)


def run_heat_control_example():
    """Example: Heat equation control"""
    print("=" * 50)
    print("HEAT EQUATION CONTROL")
    print("=" * 50)
    
    # Configuration
    config = PDEControlConfig(
        model_path="trained_models/heat1d/ckpt_ep0180.pt",
        num_epochs=2000,  # Reduced for demo
        learning_rate=1e-3,
        physics_weight=5.0,
        control_weight=1e-3
    )
    
    # Create solver
    solver = HeatControlSolver(config)
    
    # Create target profile
    target_profiles = create_target_profiles()
    x_np = np.linspace(0, 1, config.nx)
    target = target_profiles['gaussian'](x_np)
    
    # Solve control problem
    results = solver.solve(target)
    
    # Plot results
    solver.plot_results(results, target, "heat_control_results.png")
    
    print(f"Final objective value: {results['analytics']['obj'][-1]:.6f}")
    print(f"Final physics loss: {results['analytics']['physics_loss'][-1]:.6f}")
    
    return results


def run_diffusion_reaction_control_example():
    """Example: Diffusion-reaction equation control"""
    print("=" * 50)
    print("DIFFUSION-REACTION EQUATION CONTROL")
    print("=" * 50)
    
    # Configuration
    config = PDEControlConfig(
        model_path="trained_models/diffusion1d/ckpt_ep0300.pt",
        num_epochs=2000,  # Reduced for demo
        learning_rate=1e-3,
        physics_weight=5.0,
        control_weight=1e-3,
        nu=0.01,
        alpha=0.01
    )
    
    # Create solver
    solver = DiffusionReactionControlSolver(config)
    
    # Create target profile
    target_profiles = create_target_profiles()
    x_np = np.linspace(0, 1, config.nx)
    target = target_profiles['double_gaussian'](x_np)
    
    # Solve control problem
    results = solver.solve(target)
    
    # Plot results
    solver.plot_results(results, target, "diffusion_control_results.png")
    
    print(f"Final objective value: {results['analytics']['obj'][-1]:.6f}")
    print(f"Final physics loss: {results['analytics']['physics_loss'][-1]:.6f}")
    
    return results


def run_burgers_control_example():
    """Example: Burgers equation control"""
    print("=" * 50)
    print("BURGERS EQUATION CONTROL")
    print("=" * 50)
    
    # Configuration
    config = PDEControlConfig(
        model_path="trained_models/burgers1d/best.pt",
        num_epochs=2000,  # Reduced for demo
        learning_rate=1e-3,
        physics_weight=5.0,
        control_weight=1e-3,
        nu=0.01
    )
    
    # Create solver
    solver = BurgersControlSolver(config)
    
    # Create target profile
    target_profiles = create_target_profiles()
    x_np = np.linspace(0, 1, config.nx)
    target = target_profiles['shock'](x_np)
    
    # Solve control problem
    results = solver.solve(target)
    
    # Plot results
    solver.plot_results(results, target, "burgers_control_results.png")
    
    print(f"Final objective value: {results['analytics']['obj'][-1]:.6f}")
    print(f"Final physics loss: {results['analytics']['physics_loss'][-1]:.6f}")
    
    return results


def compare_solvers():
    """Compare different PDE solvers with the same target"""
    print("=" * 50)
    print("COMPARING DIFFERENT PDE SOLVERS")
    print("=" * 50)
    
    # Use same target for all solvers
    target_profiles = create_target_profiles()
    x_np = np.linspace(0, 1, 64)
    target = target_profiles['gaussian'](x_np)
    
    # Common configuration
    base_config = PDEControlConfig(
        num_epochs=1000,  # Reduced for comparison
        learning_rate=1e-3,
        physics_weight=5.0,
        control_weight=1e-3
    )
    
    # Heat equation
    heat_config = PDEControlConfig(
        model_path="trained_models/heat1d/ckpt_ep0180.pt",
        **{k: v for k, v in base_config.__dict__.items() if k != 'model_path'}
    )
    heat_solver = HeatControlSolver(heat_config)
    heat_results = heat_solver.solve(target)
    
    # Diffusion-reaction
    diff_config = PDEControlConfig(
        model_path="trained_models/diffusion1d/ckpt_ep0300.pt",
        **{k: v for k, v in base_config.__dict__.items() if k != 'model_path'}
    )
    diff_solver = DiffusionReactionControlSolver(diff_config)
    diff_results = diff_solver.solve(target)
    
    # Print comparison
    print("\nComparison of final losses:")
    print(f"Heat equation - Objective: {heat_results['analytics']['obj'][-1]:.6f}, Physics: {heat_results['analytics']['physics_loss'][-1]:.6f}")
    print(f"Diffusion-reaction - Objective: {diff_results['analytics']['obj'][-1]:.6f}, Physics: {diff_results['analytics']['physics_loss'][-1]:.6f}")
    
    return heat_results, diff_results


def main():
    """Run all examples"""
    print("PDE Control Solver Examples")
    print("This script demonstrates the structured approach to solving PDE control problems.")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Run individual examples
        heat_results = run_heat_control_example()
        print("\n" + "="*50 + "\n")
        
        diff_results = run_diffusion_reaction_control_example()
        print("\n" + "="*50 + "\n")
        
        burgers_results = run_burgers_control_example()
        print("\n" + "="*50 + "\n")
        
        # Compare solvers
        compare_solvers()
        
        print("\nAll examples completed successfully!")
        print("Check the generated PNG files for visualization of results.")
        
    except FileNotFoundError as e:
        print(f"Error: Model file not found. Please ensure the trained models exist: {e}")
    except Exception as e:
        print(f"Error during execution: {e}")


if __name__ == "__main__":
    main() 