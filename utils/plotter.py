import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import NullLocator

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

def plot_analytics(losses, epoch, last_timestamp, folder):
    """
    Plot training and validation losses over epochs.
    """
    def plot_train_test(train_loss, test_loss, path, epoch, last_timestamp):
        path = os.path.join(path, "train_test")
        os.makedirs(path, exist_ok=True)
        epochs = np.arange(1, len(train_loss) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss,
                label=f'Train (final={train_loss[-1]:.4e})',
                linestyle='-', linewidth=1, marker='')
        ax.plot(epochs, test_loss,
                label=f'Validation  (final={test_loss[-1]:.4e})',
                linestyle='-', linewidth=1, marker='')

        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Train vs. Validation Loss')
        ax.grid(which='both', ls='--', lw=0.5)
        ax.legend(loc='upper right')

        # --- exactly 5 evenly spaced ticks ---
        num_epochs = len(epochs)
        # if fewer than 5 epochs, just show them all
        if num_epochs >= 5:
            ticks = np.linspace(1, num_epochs, num=5, dtype=int)
        else:
            ticks = epochs
        ax.set_xticks(ticks)
        ax.xaxis.set_minor_locator(NullLocator())

        plt.tight_layout()
        plt.savefig(
            os.path.join(path,
                        f'epochs_{epoch}_{last_timestamp}_losses.png'),
            dpi=300
        )

    def plot_physics_initial(physics_loss, initial_loss, path, epoch, last_timestamp):
        path = os.path.join(path, "physics_initial")
        os.makedirs(path, exist_ok=True)
        epochs = np.arange(1, len(physics_loss) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, initial_loss,
                label=f'Initial (final={initial_loss[-1]:.4e})',
                linestyle='-', linewidth=1, marker='')
        ax.plot(epochs, physics_loss,
                label=f'Physics (final={physics_loss[-1]:.4e})',
                linestyle='-',linewidth=1, marker='')

        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Residual (log scale)')
        ax.set_title('Physics & Initial Loss')
        ax.grid(which='both', ls='--', lw=0.5)
        ax.legend(loc='upper right')

        # --- exactly 5 evenly spaced ticks ---
        num_epochs = len(epochs)
        if num_epochs >= 5:
            ticks = np.linspace(1, num_epochs, num=5, dtype=int)
        else:
            ticks = epochs
        ax.set_xticks(ticks)
        ax.xaxis.set_minor_locator(NullLocator())

        plt.tight_layout()
        plt.savefig(
            os.path.join(path,
                        f'epochs_{epoch}_{last_timestamp}_loss.png'),
            dpi=300
        )
   
    os.makedirs(folder, exist_ok=True)

    plot_train_test(losses['train_loss'], losses['test_loss'], folder, epoch, last_timestamp)
    plot_physics_initial(losses['physics_loss'], losses['initial_loss'], folder, epoch, last_timestamp)


def plot_optimal_vs_predicted(t, u_pred, x_pred_solution, x_pred, u_true, x_true, x_2, title=None, savepath=None):
    """
    Plots predicted vs true optimal control and state trajectories (using provided true vectors).
    """
    x_pred_np = x_pred.detach().cpu().squeeze().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    axs[0].plot(t, u_true, label='Optimal $u^*(t)$', color='black', linewidth=2)
    axs[0].plot(t, u_pred, '--', label='Predicted $u(t)$', color='tab:blue')
    axs[0].set_ylabel('Control $u(t)$')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(t, x_true, label='Optimal $x^*(t)$', color='black', linewidth=2)
    axs[1].plot(t, x_pred_np, '--', label='Predicted $x(t)$', color='tab:orange')
    axs[1].plot(t, x_pred_solution, '--', label='Solver of found u is $x(t)$', color='tab:blue')
    axs[1].plot(t, x_2, '--', label='Solver of optimal solution', color='tab:red')
    
    axs[1].set_xlabel('Time $t$')
    axs[1].set_ylabel('State $x(t)$')
    axs[1].legend()
    axs[1].grid(True)

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)

    if savepath:
        plt.savefig(savepath, dpi=200)
        print(f"Plot saved to {savepath}")
    plt.show()
