import numpy as np
import matplotlib.pyplot as plt
import os

def plot_validation_samples(data, epoch, folder, name, grid_rows=4, grid_cols=3):
    """
    Plot selected samples from validation dataset showing control functions and trajectories.
    If worst_data is provided, it should contain 'trajectory', 'u', 'predicted' lists.
    """
    # Create subplot grid
    if not os.path.exists(folder):
        os.makedirs(folder)

    name = os.path.join(folder, name)
    
    # Determine title based on available data
    title = f'Top 12 Worst Predictions - Epoch {epoch}\nControl Functions (u), True Trajectories (x), and Predictions (x_pred)'
    
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 20))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)    
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    # Time domain
    t = np.linspace(0, 1, 200)

    for i, (idx, u, x, x_pred, error) in enumerate(zip(data['indices'], data['u'], data['trajectory'], data['predicted'], data['errors'])):
            
        ax = axes[i]
        
        # Plot all three
        ax.plot(t, u, 'b-', linewidth=2, label='Control u(t)')
        ax.plot(t, x, 'r-', linewidth=2, label='True trajectory x(t)')
        ax.plot(t, x_pred, 'g--', linewidth=2, label='Prediction x_pred(t)')
     
        ax.set_title(f'Sample {idx}, Error: {error:.4f}')
        
        # Set y-limits
        all_values = np.concatenate([u, x])
    
        # Customize
        ax.set_xlabel('Time t')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Adjust layout
    plt.subplots_adjust(
        right=0.95,
        top=0.94,        # More space for title at top
        bottom=0.05,
        left=0.1,
        wspace=0.2,
        hspace=0.3
    )

    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory