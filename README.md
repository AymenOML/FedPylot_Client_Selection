## 📌 Goals and Roadmap

This project is structured around building a smart client selection mechanism in Federated Learning (FL), starting with rule-based heuristics and progressively moving to ML-based models. The implementation and testing are based on the CIFAR-10 dataset using FedPylot as the underlying simulation engine.

---

### ✅ Phase 1 – Data Preparation & Baseline Setup *(Completed)*

This foundational phase focuses on preparing the data and simulation inputs:

- [x] Scrape system utility data (e.g., FLOPS, training time) from the AI Benchmark ranking site.
- [x] Parse real Android usage traces, extracting durations for screen activity and charging as proxies for energy/battery usage.
- [x] Compute KL Divergence per client to quantify label distribution skew (non-IID degree) from CIFAR-10.
- [x] Simulate non-IID local datasets using a Dirichlet distribution (controlled with alpha) across clients.
- [x] Merge all data sources into a single structured dataset per client with features: device specs, energy usage, data size, variance, etc.

---

### 🚧 Phase 1.5 – Iterative Rule-Based Client Selector *(In Progress)*

This will serve as a baseline before training any ML-based client selection models.

- [ ] Implement an iterative rule-based client selection algorithm that:
    - Prioritizes clients with better FLOPS, lower training time, balanced battery usage (based on screen/charge patterns).
    - Includes data-based metrics like local dataset size and KL divergence (variance).
- [ ] Design a ranking or scoring function to pick the top-K clients each round.
- [ ] Integrate a participation count penalty to avoid selecting the same clients repeatedly (battery-aware).
- [ ] Compare performance vs FedAvg (random selection):
    - Global accuracy over rounds
    - Training time per round
    - Fairness in client participation
- [ ] Visualize client selection distributions across rounds.
- [ ] Log client selection history for future use in imitation learning (Phase 2).

---

### 🧠 Phase 2 – ML-Based Client Selection *(Planned)*

This phase introduces learning-based models for smarter and adaptive selection:

- [ ] Design and implement a shallow MLP that predicts a selection score per client.
- [ ] Train the MLP using imitation learning, starting with the decisions made by the rule-based selector.
- [ ] Integrate the model into FedPylot, replacing the existing client sampling function.
- [ ] Evaluate impact on:
    - Global model accuracy
    - Convergence speed
    - Fairness and diversity in client participation
- [ ] Extend the model to be battery-aware, either through adjusted input features or via a regularized loss.
