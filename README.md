# Neural Quantum States: J1-J2 Heisenberg Model with CNNs

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

> A machine learning project at the intersection of deep learning and quantum many-body physics.  
> We use 1D Convolutional Neural Networks within a Variational Monte Carlo (VMC) framework to approximate ground states of frustrated quantum spin chains.

**Course:** Math 156 – Machine Learning (Fall 2025, UCLA)  

## Contributions

**Supervised CNN baseline** — the core model that validates the neural network architecture before deploying the full unsupervised VMC system.

- **Physics-informed input engineering:** Designed a 3-channel `(3 × 14)` input tensor that embeds spin configurations alongside J1 and J2 coupling constants, enabling the model to learn local energy contributions tied to specific Hamiltonian parameters
- **CNN architecture design:** Built a 1D-CNN with two convolutional layers + Global Average Pooling + MLP regression head, achieving translational invariance consistent with the physics of spin chains
- **Dataset pipeline:** Generated 73,710 configurations via Exact Diagonalization across 9 (J1, J2) parameter pairs; implemented stratified 70/15/15 train-val-test splits
- **Training & evaluation:** Trained with Adam optimizer (MSE loss), achieving a test RMSE of **0.1399** — a 24.4× improvement over baseline, capturing **99.8% of energy variance**
- **Validated the architecture** as a suitable ansatz for the subsequent unsupervised VMC phase
- Also contributed to **unsupervised CNN training logic and debugging**

---

## 🌌 Project Overview

Simulating quantum systems is computationally hard because the state space grows exponentially with system size (2^N). This "Curse of Dimensionality" blocks exact simulation beyond ~40 particles.

This project uses **Neural Quantum States (NQS)** to compress the wavefunction into a neural network. We focus on the **1D J1–J2 Heisenberg Model**, a system with geometric frustration and complex phase transitions.

We implement and compare three approaches:

| Approach | Description |
|---|---|
| **Exact Diagonalization** | Brute-force physics engine; generates ground truth labels (limited to small N) |
| **Supervised CNN** | CNN trained to regress energy from spin configurations + coupling parameters |
| **Unsupervised VMC** | CNN trained via RL-style optimization to minimize ⟨H⟩ directly, no labels |

---

## 📊 Key Results

### Supervised Baseline
- Test RMSE: **0.1399** vs dataset std of 3.42 → **24.4× improvement**
- R² ≈ **0.998** — CNN captures nearly all variance in ground-state energy
- Validation loss tracked training loss closely, stabilizing within 5 epochs

### Unsupervised VMC (N=8, J2=0.5)
- **Real CNN** (with Marshall Sign Rule): fast convergence but plateaus above exact energy
- **Complex CNN**: overcomes the sign problem by learning phase angles autonomously; closes the gap by epoch 350
- Key finding: kernel size k=5 outperforms k=3 — the network learns non-local correlations beyond nearest neighbors

### Large System (N=16)
- Complex CNN successfully trained on a system too large for exact diagonalization
- Spin texture stabilizes early into a dimerized pattern consistent with the Majumdar-Ghosh phase

---

## 📂 Repository Structure

```
.
├── models/
│   ├── cnn.py                  # Real-valued 1D CNN
│   └── complex_cnn.py          # Complex-valued CNN (log-amplitude + phase heads)
├── physics/
│   ├── j1j2_solver.py          # Exact Diagonalization solver
│   └── j1j2_unsupervised.py    # Local energy + Marshall Sign Rule logic
├── train/
│   ├── train_cnn.py            # Supervised training loop
│   └── train_vmc.py            # Unsupervised VMC optimization loop
├── utils/
│   ├── mcmc_sampler.py         # Metropolis-Hastings MCMC sampler
│   └── plotting.py             # Visualization tools
├── datasets/                   # Generated spin configuration datasets
├── plots/                      # Output figures
├── Supervised_CNN.ipynb        # Supervised baseline (my primary contribution)
├── run_unsupervised.ipynb      # Full VMC experiment
├── hyperparam_tune.ipynb       # Grid search over architecture hyperparameters
├── large_n_system.ipynb        # Scalability experiments (N=16)
├── eda_analysis_stratified.ipynb  # Exploratory data analysis
└── README.md
```

---

## 🧠 Model Architectures

### Supervised CNN (Physics-Informed Input)
```
Input: (3 × 14) tensor
  - Channel 0: spin configuration σ ∈ {-1, 0, +1}
  - Channel 1: J1 coupling (broadcasted)
  - Channel 2: J2 coupling (broadcasted)

Conv1D(16 filters, k=3) → ReLU
Conv1D(32 filters, k=3) → ReLU
Global Average Pooling  ← enforces translational invariance
MLP → scalar E₀
```

### Unsupervised Complex CNN (VMC Ansatz)
```
Input: spin configuration
Conv1D → ELU → Conv1D → Global Avg Pool
→ Log-Amplitude head: outputs ln|A|
→ Phase head:          outputs φ
→ Ψ(S) = e^(ln|A| + iφ)
```
The complex architecture solves the **sign problem** in frustrated systems by learning phase angles autonomously, without hard-coded physical priors.

---

## 🚀 Getting Started

```bash
git clone https://github.com/qyunhan/j1j2model.git
cd j1j2model
pip install torch numpy scipy matplotlib jupyter
```

**Run the supervised baseline:**
```bash
jupyter notebook Supervised_CNN.ipynb
```

**Run the full VMC experiment:**
```bash
jupyter notebook run_unsupervised.ipynb
```

**Generate ground truth data (small systems):**
```python
from physics.j1j2_solver import QuantumJ1J2Solver
solver = QuantumJ1J2Solver(n_spins=10, J1=1.0, J2=0.5)
print(f"Exact Ground State Energy: {solver.ground_state_energy}")
```

---

## 📚 References

1. Liang, X. et al. (2018). *Solving frustrated quantum many-particle models with convolutional neural networks.* Physical Review B, 98(10). https://doi.org/10.1103/physrevb.98.104426
2. Lange, H. et al. (2024). *From architectures to applications: A review of Neural Quantum States.* Quantum Science and Technology, 9(4). https://doi.org/10.1088/2058-9565/ad7168
3. Carleo, G., & Troyer, M. (2017). *Solving the quantum many-body problem with artificial neural networks.* Science.
