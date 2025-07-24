# Thesis: Operator Learning and Optimal Control

This repository is part of a thesis project exploring how different neural operator architectures can be trained and applied to optimal control problems. It provides implementations, training scripts, and usage examples for DeepONet, FNO, and LNO architectures, as well as utilities for dataset generation, training, and evaluation.

## Project Structure

```
Thesis-2/
├── models/              # Model architectures: DeepONet, FNO, LNO
├── trained_models/      # Saved weights of trained models
├── utils/               # Utilities for data, training, plotting, metrics, etc.
├── examples/            # Example scripts for training, dataset generation, and control problems
├── analytics_plots/     # Analytics and loss plots
├── plots/               # Prediction and validation plots
├── found_trajectories/  # Plots of optimal vs predicted trajectories
├── playground_deeponet.ipynb   # DeepONet explanatory notebook
├── playground_FNO.ipynb        # FNO explanatory notebook
├── playground_LNO.ipynb        # LNO explanatory notebook
├── README.md            # Project overview and instructions
└── requirements.txt     # Python dependencies
```

## Datasets

The datasets used for training neural operators on various ODE control problems (linear, nonlinear, oscillatory, polynomial tracking, singular arc) are available for download here:

- [Google Drive: Datasets](https://drive.google.com/drive/folders/1vEzBSFHIj2BYLJh-PqvZWgMwNrMM0MLO?usp=sharing)

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
- Use the scripts in `examples/` to train a model:
  - `training_deep_o_net.py` — Train DeepONet
  - `training_fno.py` — Train FNO
  - `training_lno.py` — Train LNO
- You can also use the Jupyter notebooks (`playground_deeponet.ipynb`, `playground_FNO.ipynb`, `playground_LNO.ipynb`) for interactive exploration, training, and visualization.

### Generating Datasets
- Use `examples/generate_dataset.py` to generate datasets for different ODE control problems. The script supports various function types and ODEs (linear, oscillatory, polynomial tracking, nonlinear, singular arc).

### Solving Control Problems
- `examples/control_problem.py` demonstrates how to load a trained model and solve an optimal control problem using the provided neural operator.

### Utilities
- The `utils/` directory contains:
  - `data.py`: Dataset classes and ODE problem registry
  - `training.py`: Training loop and early stopping
  - `metrics.py`: Error and metric calculations
  - `plotter.py`: Plotting functions for analytics and predictions
  - `scripts.py`: Data loading, optimization routines, and helpers
  - `settings.py`: Predefined datasets, objective functions, and problem settings

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
