# architecture.md

## Fabric Intelligence Model (FIM) Architecture

The Fabric Intelligence Model operates on a latent field representation that evolves over time under learned operators combining propagation, nonlinearity, memory, and retrieval.

### 1. Input Encoding

Observations \( x_t \in \mathbb{R}^{B \times C \times H \times W} \) are mapped into a latent fabric:

\[
z_t = E_\theta(x_t, u_t)
\]

where \( u_t \) is optional control input. The encoder is a convolutional residual network producing latent channels \( z_t \in \mathbb{R}^{B \times C_f \times H \times W} \).

---

### 2. Latent Fabric State

The model maintains a latent field:

\[
F_t \in \mathbb{R}^{B \times C_f \times H \times W}
\]

This field encodes spatial, temporal, and memory-dependent structure.

---

### 3. Write Operator

New information is injected into the fabric via a gated write:

\[
F_t^{+} = F_t + \sigma(W_g z_t) \odot \tanh(W_p z_t)
\]

This allows selective updates based on current input salience.

---

### 4. Evolution Operator

The fabric evolves according to:

\[
F_{t+1}^{\text{base}} = F_t^{+} + \Delta t \cdot \left( P_\phi(F_t^{+}) + N_\phi(F_t^{+}) - \lambda F_t^{+} \right)
\]

where:

- \( P_\phi \): propagation operator (depthwise diffusion + mixing)
- \( N_\phi \): nonlinear interaction (MLP-style convolution)
- \( \lambda \): learnable damping

---

### 5. Retrieval Injection

Optional memory retrieval modifies the state:

\[
F_{t+1} = F_{t+1}^{\text{base}} + W_r(R_t)
\]

where \( R_t \in \mathbb{R}^{B \times C_f} \) is broadcast spatially.

---

### 6. Residual Mixing

Stabilization is achieved via:

\[
F_{t+1}^{\text{final}} = W_m([F_t^{+}, F_{t+1}])
\]

---

### 7. Decoder

Predictions are obtained via:

\[
\hat{y}_t = D_\psi(F_t)
\]

mapping latent fabric back to observation space.

---

### 8. Memory System

A sparse memory stores salient states:

\[
M = \{(k_i, v_i)\}_{i=1}^K
\]

Retrieval is performed via similarity in latent space.

---

### 9. Full Update

\[
F_{t+1} = \mathcal{F}(F_t, x_t, R_t)
\]

\[
\hat{y}_t = D(F_t)
\]

---

### 10. Key Properties

- Spatially structured latent dynamics
- Continuous-time inspired evolution
- Memory-augmented updates
- Stable long-horizon rollouts