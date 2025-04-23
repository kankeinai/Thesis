# Thesis: Operator Learning and Optimal Control

Created as part of a thesis project to explore how different arcitectures of neural operators can be trained and used in optimal control problems

## Structure

```
/project-root
├── /models/           # Source code of models architecture
├── /trained_models/          # Weights of trained models
├── /utils/         # Scripts used in the training, plotting
├── playground.ipynb        # Deep-O-Net explanatory notebook
├── playground_FNO.ipynb         # FNO explanatory notebook
├── playground_LNO.ipynb         # LNO explanatory notebook
├── training_FNO.ipynb         # Script to train FNO
├── training_LNO.ipynb         # Script to train LNO
└── README.md       # Project overview and instructions

```

## Installation

Follow these steps to set up the project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/kankeinai/Thesis
    cd your-repo
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. Run `training_[model].py` or `playground_[model].ipynb`

## Contributing

Feel free to submit issues or pull requests to improve the project.

## License

This project is licensed under the [MIT License](LICENSE).
