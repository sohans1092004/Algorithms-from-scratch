# Algorithms-from-scratch

A hands-on collection of classic Machine Learning and Deep Learning algorithms implemented from first principles in Python/Jupyter.

The goal of this repository is to deepen understanding of core ML/DL techniques by building them “from scratch,” without relying on high-level libraries.

---

## 📖 Table of Contents

- [About](#about)
- [Motivation](#motivation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebooks](#running-the-notebooks)
- [Implemented Algorithms](#implemented-algorithms)
- [Contributing](#contributing)
- [License](#license)
---

## About

This project hosts step-by-step implementations of foundational ML and DL methods, including data preprocessing routines, loss functions, performance metrics, and optimization algorithms. Each algorithm is presented in its own notebook or module so you can follow the math, code it yourself, and visualize the results.

---

## Motivation

Understanding how ML/DL algorithms work “under the hood” is crucial for:

- **Gaining intuition** about model behavior
- **Debugging and customizing** algorithms
- **Appreciating the trade-offs** in convergence, complexity, and generalization

By coding everything from scratch, you don’t just call `fit()`—you derive gradients, implement your own solvers, and see every intermediate step.

---

### Repository Structure

-   **ML algorithms/**: Classic “shallow” learning models
    -   `linear_regression.ipynb`
    -   `logistic_regression.ipynb`
    -   `decision_tree.ipynb`
    -   `k_nearest_neighbors.ipynb`
-   **DL algorithms/**: Neural network architectures
    -   `perceptron.ipynb`
    -   `mlp_from_scratch.ipynb`
    -   `convolutional_nn.ipynb`
    -   `rnn_from_scratch.ipynb`
-   **preprocessing/**: Data preparation utilities
    -   `normalization.py`
    -   `train_test_split.py`
    -   `one_hot_encoding.py`
-   **losses/**: Loss function implementations
    -   `mean_squared_error.py`
    -   `cross_entropy.py`
-   **metrics/**: Evaluation metrics
    -   `accuracy.py`
    -   `precision_recall.py`
    -   `roc_auc.py`
-   **optimizers/**: Gradient-based optimizers
    -   `sgd.py`
    -   `momentum.py`
    -   `rmsprop.py`
    -   `adam.py`
-   **utils/**: Helper functions
    -   `visualize.py`
-   `README.md`

---

## Getting Started

### Prerequisites

- Python 3.7+
- [Jupyter Notebook](https://jupyter.org/)
- The following Python packages:
  ```bash
  numpy
  pandas
  matplotlib
  scipy

## Installation

### 1. Clone this repo
```bash
git clone https://github.com/sohans1092004/Algorithms-from-scratch.git
cd Algorithms-from-scratch
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, simply install the packages listed above.

## Running the Notebooks

Simply launch Jupyter in the root directory and open any notebook:

```bash
jupyter notebook
```

Select the notebook you want (e.g., `ML algorithms/linear_regression.ipynb`) and run cell by cell to see the derivations, code, and plots.

## Implemented Algorithms

### Machine Learning

* **Linear Regression**: Ordinary Least Squares solution; gradient descent version
* **Logistic Regression**: Binary classification via sigmoid activation
* **Decision Tree**: ID3/CART splitting on entropy or Gini
* **k-Nearest Neighbors**: Brute-force distance comparisons

### Deep Learning

* **Perceptron**: Single-layer binary classifier
* **Multi-Layer Perceptron (MLP)**: Fully connected neural net with backpropagation
* **Convolutional Neural Network (CNN)**: Convolutions, pooling, and dense layers
* **Recurrent Neural Network (RNN)**: Vanilla RNN cell with sequence processing

### Utilities

* **Preprocessing**: normalization, one-hot encoding, train/test split
* **Losses**: MSE, cross-entropy
* **Metrics**: accuracy, precision, recall, ROC-AUC
* **Optimizers**: SGD, Momentum, RMSProp, Adam

## Contributing

Contributions, enhancements, and new algorithms are very welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-algo`)
3. Commit your changes (`git commit -m "Add X algorithm"`)
4. Push to origin (`git push origin feature/your-algo`)
5. Open a Pull Request

## License
This project is licensed under the MIT License. 
