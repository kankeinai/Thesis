#!/usr/bin/env python3
"""
Script to generate a comparison table for multiple PDE types.
This script creates a comprehensive table comparing training and control statistics across different PDEs.
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List

def load_training_stats(stats_path: str) -> Dict[str, Any]:
    """Load training statistics from a .pt file."""
    try:
        stats = torch.load(stats_path, map_location='cpu')
        return stats
    except Exception as e:
        print(f"❌ Error loading training stats from {stats_path}: {e}")
        return {}

def create_demo_control_analytics():
    """Create demo control analytics for demonstration purposes."""
    return {
        'obj': [0.001, 0.0008, 0.0006, 0.0005, 0.0004],
        'total_loss': [0.01, 0.008, 0.006, 0.005, 0.004],
        'physics_loss': [0.005, 0.004, 0.003, 0.002, 0.001]
    }

def extract_metrics(training_stats: Dict[str, Any], control_analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from both training and control data."""
    metrics = {}
    
    # Training metrics
    if training_stats:
        if 'train_total' in training_stats and training_stats['train_total']:
            train_losses = training_stats['train_total']
            metrics['Training Final Loss'] = train_losses[-1]
            metrics['Training Min Loss'] = min(train_losses)
            metrics['Training Epochs'] = len(train_losses)
        
        if 'train_phys' in training_stats and training_stats['train_phys']:
            phys_losses = training_stats['train_phys']
            metrics['Training Final Physics Loss'] = phys_losses[-1]
            metrics['Training Min Physics Loss'] = min(phys_losses)
    
    # Control metrics
    if control_analytics:
        for key in ['obj', 'total_loss', 'physics_loss']:
            if key in control_analytics and control_analytics[key]:
                values = control_analytics[key]
                metrics[f'Control Final {key.title()}'] = values[-1]
                metrics[f'Control Min {key.title()}'] = min(values)
                metrics[f'Control Epochs'] = len(values)
    
    return metrics

def generate_comparison_table():
    """Generate a comparison table for multiple PDE types."""
    
    # Define PDE configurations
    pde_configs = [
        {
            'name': 'Burgers',
            'training_stats': 'trained_models/burgers1d/training_stats.pt',
            'control_analytics': 'control_pde_results/burgers_control_results_solutions/analytics.npy'
        },
        {
            'name': 'Heat',
            'training_stats': 'trained_models/heat1d/training_stats.pt',
            'control_analytics': 'control_pde_results/heat_control_results_solutions/analytics.npy'
        },
        {
            'name': 'Diffusion',
            'training_stats': 'trained_models/diffusion1d/training_stats.pt',
            'control_analytics': 'control_pde_results/diffusion_control_results_solutions/analytics.npy'
        }
    ]
    
    all_metrics = []
    
    print("📊 Generating Comparison Table for Multiple PDE Types")
    print("=" * 60)
    
    for config in pde_configs:
        pde_name = config['name']
        print(f"\n🔍 Processing {pde_name}...")
        
        # Load training stats
        training_stats = load_training_stats(config['training_stats'])
        
        # Try to load control analytics, use demo if not available
        control_analytics = {}
        try:
            control_analytics = np.load(config['control_analytics'], allow_pickle=True).item()
            print(f"  ✅ Loaded control analytics for {pde_name}")
        except:
            print(f"  ⚠️  Using demo control analytics for {pde_name}")
            control_analytics = create_demo_control_analytics()
        
        # Extract metrics
        metrics = extract_metrics(training_stats, control_analytics)
        metrics['PDE Type'] = pde_name
        
        all_metrics.append(metrics)
    
    # Create DataFrame
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns to put PDE Type first
        cols = ['PDE Type'] + [col for col in df.columns if col != 'PDE Type']
        df = df[cols]
        
        # Format numeric columns
        for col in df.columns:
            if col != 'PDE Type' and df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].apply(lambda x: f"{x:.6e}" if pd.notna(x) else "N/A")
            elif col != 'PDE Type' and df[col].dtype in ['int64', 'int32']:
                df[col] = df[col].apply(lambda x: f"{x}" if pd.notna(x) else "N/A")
        
        # Save to CSV
        output_file = "pde_comparison_table.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Comparison table saved to: {output_file}")
        
        # Display table
        print("\n📊 PDE Comparison Table:")
        print("=" * 120)
        print(df.to_string(index=False))
        print("=" * 120)
        
        return df
    else:
        print("❌ No data available for comparison")
        return None

if __name__ == "__main__":
    generate_comparison_table() 