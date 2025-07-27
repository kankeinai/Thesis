# Thesis: Operator Learning and Optimal Control

This repository is part of a thesis project exploring how different neural operator architectures can be trained and applied to optimal control problems. It provides implementations, training scripts, and usage examples for DeepONet, FNO, and LNO architectures, as well as utilities for dataset generation, training, and evaluation.

## Project Structure

```
Thesis/
├── models/              # Model architectures: DeepONet, FNO, LNO
├── trained_models/      # Saved weights and checkpoints of trained models
├── utils/               # Utilities for data, training, plotting, metrics, settings, etc.
├── scripts/             # Scripts for training, dataset generation, control, and experiments
├── analytics_plots/     # Analytics and loss plots for training runs
├── computation_time/    # Timing logs for model training
├── plots/               # Prediction and validation plots
├── found_trajectories/  # Plots of optimal vs predicted trajectories
├── results/             # CSV result files (e.g., sensitivity studies)
├── tables/              # LaTeX tables for results
├── playground_deeponet.ipynb   # DeepONet explanatory notebook
├── playground_FNO.ipynb        # FNO explanatory notebook
├── playground_LNO.ipynb        # LNO explanatory notebook
├── baseline.py          # Baseline optimal control solutions (CasADi)
├── requirements.txt     # Python dependencies
├── LICENSE              # License file
└── README.md            # Project overview and instructions
```

## Datasets

The datasets used for training neural operators on various ODE control problems (linear, nonlinear, oscillatory, polynomial tracking, singular arc) are available for download here:

- [Google Drive: Datasets](https://drive.google.com/drive/folders/1vEzBSFHIj2BYLJh-PqvZWgMwNrMM0MLO?usp=sharing)

You can also generate datasets using the script:
- `scripts/generate_dataset.py` — Flexible dataset generation for all supported ODE problems and function types.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kankeinai/Thesis
   cd Thesis-2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model
- Use the scripts in `scripts/` to train a model:
  - `training_deep_o_net.py` — Train DeepONet
  - `training_fno.py` — Train FNO
  - `training_lno.py` — Train LNO
- You can also use the Jupyter notebooks (`playground_deeponet.ipynb`, `playground_FNO.ipynb`, `playground_LNO.ipynb`) for interactive exploration, training, and visualization.

### Generating Datasets
- Use `scripts/generate_dataset.py` to generate datasets for different ODE control problems. The script supports various function types and ODEs (linear, oscillatory, polynomial tracking, nonlinear, singular arc).

### Solving Control Problems
- `scripts/control_problem.py` demonstrates how to load a trained model and solve an optimal control problem using the provided neural operator.

### Running Experiments and Analysis
- `scripts/run_rho_sensitivity.py` — Run a sensitivity study on the control weight parameter ρ across all architectures and benchmarks. Results are saved to `results/rho_sensitivity.csv` and `results/rho_sensitivity_summary.csv`.
- `scripts/computation_time.py` — Analyze and print computation times for model training (uses files in `computation_time/`).
- `scripts/make_rho_table.py` — Generate LaTeX tables summarizing results (output in `tables/`).

### Baseline Solutions
- `baseline.py` — Provides CasADi-based optimal control solutions for all benchmark problems. Outputs are saved in `baseline_solutions/` as `.npz` files.

### Utilities
- The `utils/` directory contains:
  - `data.py`: Dataset classes, ODE problem registry, and data utilities
  - `training.py`: Training loop, early stopping, and checkpointing
  - `metrics.py`: Error and metric calculations
  - `plotter.py`: Plotting functions for analytics and predictions
  - `scripts.py`: Data loading, optimization routines, and helpers
  - `settings.py`: Predefined datasets, objective functions, and problem settings

## Outputs and Results
- **Analytics and Loss Plots:**
  - `analytics_plots/` — Training analytics and loss curves for all experiments
- **Prediction and Validation Plots:**
  - `plots/` — Model predictions vs. ground truth for test sets
- **Found Trajectories:**
  - `found_trajectories/` — Plots comparing optimal and predicted trajectories
- **Computation Time:**
  - `computation_time/` — Timing logs for model training (per architecture/problem)
- **Results:**
  - `results/rho_sensitivity.csv` — Raw results from ρ sensitivity study
  - `results/rho_sensitivity_summary.csv` — Summary statistics for ρ sensitivity
- **Tables:**
  - `tables/pilot_rho_landscape.tex` — LaTeX tables for paper-ready results
- **Trained Models:**
  - `trained_models/` — All model checkpoints and best weights (organized by problem/architecture)

## Supported ODE Problems
- **linear**: \( dx/dt = -x + u,\ x(0) = 1 \)
- **oscillatory**: \( dx/dt = \cos(4\pi t) + u,\ x(0) = 0 \)
- **polynomial_tracking**: \( dx/dt = u,\ x(0) = 0 \)
- **nonlinear**: \( dx/dt = \frac{5}{2}(-x + x u - u^2),\ x(0) = 1 \)
- **singular_arc**: \( dx/dt = x^2 + u,\ x(0) = 1 \)

## Contributing

Contributions are welcome! Please submit issues or pull requests to improve the project.

## License

This project is licensed under the [MIT License](LICENSE).
