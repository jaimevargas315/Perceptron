# Perceptron Learning and Classification in Julia

This repository contains Julia implementations of the Rosenblatt Perceptron, featuring both Stochastic Gradient Descent (Online) and Batch learning approaches, along with complex synthetic dataset generators.

---

## 1. Perceptron Models

The core classification logic is based on the linear threshold unit that maps an input vector to a binary output.

* **Perceptron Activation (`perceptron`):** Implements the forward pass. It augments the input vector with a bias term ($x_0 = 1$) and applies a signum activation function based on the weight vector $\mathbf{w}$.
* **Online Perceptron Training (`trainPerceptron`):** An implementation of the Stochastic Gradient Descent (SGD) update rule. Weights are updated immediately after processing each misclassified sample:
    $$\mathbf{w} \leftarrow \mathbf{w} + \eta (d_n - y_n) \mathbf{x}_n$$
    This version includes data shuffling at the start of each epoch to ensure stochasticity.


* **Batch Perceptron Training (`trainBatchPerceptron`):** Instead of updating per sample, this version accumulates errors across the entire dataset (the "batch") and performs a single weight update per epoch. This approach provides a smoother gradient descent path toward the solution.

---

## 2. Synthetic Dataset Generators

To test the Perceptron's ability to find linear decision boundaries, the following generators are provided:

* **Double Moon (`doublemoon`):** Generates two interlocking semi-circular distributions. This is used to test the Perceptron on datasets that may or may not be linearly separable depending on the vertical separation parameter `d`.

* **XOR Gaussian (`GaussX`):** Creates a Gaussian distribution where labels are assigned based on quadrants (Q1/Q3 vs Q2/Q4). This serves as a classic counter-example for the Perceptron, as a single linear layer cannot solve the non-linear XOR problem.

---

## 3. Implementation Details

### Mathematical Framework
The weight vector $\mathbf{w}$ is of length $F+1$, where $F$ is the number of features. The first element $w_0$ represents the **bias (threshold)**.
The decision boundary is defined by the hyperplane where:
$$\mathbf{w}^T \mathbf{x} = 0$$

### Key Functions
* **`matrix_to_vecvec`:** A utility to convert standard Julia matrices into the `Vector{Vector{Float64}}` format used by the training functions.
* **Convergence:** The Batch trainer includes a convergence check that monitors the change in the weight vector norm ($\|\mathbf{w} - \mathbf{w}_{old}\| < tol$).

---

## Technical Dependencies

* **Julia v1.6+**
* **LinearAlgebra**: For `dot()` products and `norm()` calculations.
* **Random**: For `shuffle!` operations during training.
* **Plots**: For visualizing decision boundaries and data clusters.

---

## Quick Start

```julia
# 1. Generate data
X_matrix, labels = doublemoon(500, d=1.0)
X = matrix_to_vecvec(X_matrix)

# 2. Train using Online SGD
final_weights = trainPerceptron(X, labels, 0.1)

# 3. Test a single point
prediction = perceptron([5.0, 2.0], final_weights)
