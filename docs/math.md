# math.md

## Mathematical Formulation of FIM

### 1. Problem Setting

We model a dynamical system:

\[
x_{t+1} = \mathcal{T}(x_t)
\]

with observations \( x_t \in \mathbb{R}^d \).

---

### 2. Latent Representation

Define a latent field:

\[
F_t \in \mathbb{R}^{C \times H \times W}
\]

with encoder:

\[
z_t = E(x_t)
\]

---

### 3. Write Operation

\[
F_t^{+} = F_t + G(z_t)
\]

where:

\[
G(z_t) = \sigma(W_g z_t) \odot \tanh(W_p z_t)
\]

---

### 4. Evolution Dynamics

We define a discrete approximation to a continuous system:

\[
\frac{dF}{dt} = P(F) + N(F) - \lambda F
\]

Discretized as:

\[
F_{t+1} = F_t + \Delta t \cdot \left(P(F_t) + N(F_t) - \lambda F_t\right)
\]

---

### 5. Propagation Operator

\[
P(F) = K * F
\]

where \( K \) is a spatial kernel (e.g. Laplacian).

---

### 6. Nonlinear Interaction

\[
N(F) = W_2 \sigma(W_1 F)
\]

---

### 7. Retrieval

Memory produces:

\[
R_t = \sum_i \alpha_i v_i
\]

\[
\alpha_i = \text{softmax}(k_i^\top q_t)
\]

Injected as:

\[
F_t = F_t + W_r(R_t)
\]

---

### 8. Stability

Damping ensures:

\[
\|F_t\| \leq C
\]

through:

\[
-\lambda F_t
\]

---

### 9. Prediction

\[
\hat{x}_t = D(F_t)
\]

---

### 10. Training Objective

\[
\mathcal{L}_{pred} = \mathbb{E}[\|x_{t+1} - \hat{x}_{t+1}\|^2]
\]

\[
\mathcal{L}_{stab} = \mathbb{E}[\|F_t\|^2]
\]

\[
\mathcal{L}_{mem} = \mathbb{E}[\text{sparsity}(M)]
\]

\[
\mathcal{L}_{retr} = \mathbb{E}[\|R_t - R_t^*\|^2]
\]

\[
\mathcal{L}_{cons} = \mathbb{E}[\|F_t^{pre} - F_t^{post}\|^2]
\]

---

### 11. Final Objective

\[
\mathcal{L} =
\sum_i \lambda_i \mathcal{L}_i
\]

---

### 12. Interpretation

The model approximates a nonlinear operator:

\[
\mathcal{F}: (F_t, x_t, M) \rightarrow F_{t+1}
\]

learning dynamics directly in latent space.

---

### 13. Connection to Continuous Systems

The formulation corresponds to:

- Reaction-diffusion systems
- Neural operators
- Memory-augmented dynamical systems

---

### 14. Summary

FIM learns:

- latent dynamics
- structured propagation
- nonlinear interactions
- memory-augmented evolution

in a unified framework.