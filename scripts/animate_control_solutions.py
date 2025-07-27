#!/usr/bin/env python3
"""
Script to generate animations from existing PDE control solutions.
This script loads saved solutions and creates animations showing the control input and state evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
from typing import Dict, Any, Optional

def load_solution_data(solutions_dir: str) -> Dict[str, Any]:
    """
    Load solution data from a solutions directory.
    
    Args:
        solutions_dir: Path to the solutions directory
        
    Returns:
        Dictionary containing loaded data
    """
    data = {}
    
    # Load numpy files
    data['u_optimal'] = np.load(os.path.join(solutions_dir, 'u_optimal.npy'))
    data['y_optimal'] = np.load(os.path.join(solutions_dir, 'y_optimal.npy'))
    data['target_profile'] = np.load(os.path.join(solutions_dir, 'target_profile.npy'))
    
    # Load grid info with error handling
    try:
        data['grid_info'] = np.load(os.path.join(solutions_dir, 'grid_info.npy'), allow_pickle=True).item()
    except:
        # Create default grid info if loading fails
        data['grid_info'] = {
            'x_grid': np.linspace(0, 1, 64),
            't_grid': np.linspace(0, 1, 100),
            'dx': 1/63,
            'dt': 1/99
        }
    
    # Load analytics with error handling
    try:
        data['analytics'] = np.load(os.path.join(solutions_dir, 'analytics.npy'), allow_pickle=True).item()
    except:
        data['analytics'] = {}
    
    # Load config with error handling
    try:
        data['config'] = np.load(os.path.join(solutions_dir, 'config.npy'), allow_pickle=True).item()
    except:
        data['config'] = {}
    
    return data

def create_control_animation(data: Dict[str, Any], save_path: str, title: str = "PDE Control Solution"):
    """
    Create an animation from loaded solution data.
    
    Args:
        data: Dictionary containing solution data
        save_path: Path to save the animation
        title: Title for the animation
    """
    u_optimal = data['u_optimal']
    y_optimal = data['y_optimal']
    target_profile = data['target_profile']
    grid_info = data['grid_info']
    
    x_grid = grid_info['x_grid']
    t_grid = grid_info['t_grid']
    x_min, x_max = x_grid[0], x_grid[-1]
    
    # Ensure correct shapes
    if len(u_optimal.shape) == 1:
        u_optimal = u_optimal.reshape(-1, 1)
    if len(y_optimal.shape) == 3 and y_optimal.shape[0] == 1:
        y_optimal = y_optimal.squeeze(0)  # Remove batch dimension
    if len(y_optimal.shape) == 2:
        y_optimal = y_optimal.reshape(y_optimal.shape[0], y_optimal.shape[1], 1)
    if len(target_profile.shape) == 1:
        target_profile = target_profile.reshape(-1, 1)
    
    # Set up the figure
    fig, (ax_u, ax_y) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot control input (static)
    ax_u.plot(x_grid, u_optimal.flatten(), 'b-', linewidth=2, label='Optimal Control')
    ax_u.set_xlabel('x')
    ax_u.set_ylabel('u(x)')
    ax_u.set_title('Optimal Control Input')
    ax_u.grid(True)
    ax_u.legend()
    
    # Set up animation for state evolution
    ax_y.set_xlim(x_min, x_max)
    ax_y.set_xlabel('x')
    ax_y.set_ylabel('y(x,t)')
    ax_y.set_title(f'{title} - State Evolution')
    ax_y.grid(True)
    
    # Plot target profile (static)
    ax_y.plot(x_grid, target_profile.flatten(), 'r--', linewidth=2, label='Target Profile')
    
    # Initialize lines for animation
    state_line, = ax_y.plot([], [], 'b-', linewidth=2, label='Current State')
    ax_y.legend()
    
    def init():
        state_line.set_data([], [])
        return state_line,
    
    def animate(t):
        # Get current state at time t
        if len(y_optimal.shape) == 3:
            current_state = y_optimal[:, t, 0] if y_optimal.shape[2] == 1 else y_optimal[:, t]
        else:
            current_state = y_optimal[:, t]
        
        state_line.set_data(x_grid, current_state)
        
        # Update title with current time
        current_time = t_grid[t]
        ax_y.set_title(f'{title} - State Evolution (t={current_time:.2f})')
        
        return state_line,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(t_grid), 
        init_func=init, blit=True, interval=100
    )
    
    # Save as GIF
    anim.save(save_path, writer='pillow', fps=10)
    print(f"✅ Saved animation to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def main():
    parser = argparse.ArgumentParser(description='Generate animations from PDE control solutions')
    parser.add_argument('solutions_dir', help='Path to the solutions directory')
    parser.add_argument('--output', '-o', default=None, help='Output path for animation (default: auto-generated)')
    parser.add_argument('--title', '-t', default='PDE Control Solution', help='Title for the animation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.solutions_dir):
        print(f"❌ Solutions directory not found: {args.solutions_dir}")
        return
    
    # Load solution data
    print(f"Loading solution data from: {args.solutions_dir}")
    data = load_solution_data(args.solutions_dir)
    
    # Generate output path if not provided
    if args.output is None:
        base_name = os.path.basename(args.solutions_dir).replace('_solutions', '')
        args.output = f"{base_name}_animation.gif"
    
    # Create animation
    print(f"Creating animation: {args.title}")
    create_control_animation(data, args.output, args.title)

if __name__ == "__main__":
    main() 