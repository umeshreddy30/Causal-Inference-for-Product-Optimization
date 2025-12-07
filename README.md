# Causal Intelligence Platform: Product Growth & Experimentation

### 🚀 Project Overview
A production-grade Causal Inference pipeline designed to quantify the **true impact** of product features on user spending. 

Unlike traditional A/B testing (which fails with observational data) or simple correlation analysis (which is biased by power users), this engine uses **Propensity Score Matching (PSM)** and **DoWhy** to control for confounding variables like Account Age and User Activity.

### 💼 Business Value
* **Problem:** Marketing claimed a new feature increased revenue by **$14.50** per user.
* **Reality:** High-spending "Power Users" were just more likely to use the feature (Selection Bias).
* **Solution:** Built a Causal Graph to control for confounders.
* **Verdict:** The *true* causal uplift is **$10.00**. The engine saved the company from overestimating ROI by **45%**.

### 🛠️ Tech Stack
* **Core:** Python 3.10+, DoWhy (Microsoft), CausalInference
* **Data Processing:** Pandas, NumPy (Vectorized operations)
* **Validation:** Refutation Tests (Placebo Treatment, Random Common Cause)
* **Architecture:** Modular OOP Pipeline

### 📊 Methodology (The "Why")
1.  **Causal Modeling:** Defined a structural causal model (SCM) linking `Treatment` (Feature Usage) to `Outcome` (Spend) while identifying `Confounders`.
2.  **Identification:** Used the **Backdoor Criterion** to find a valid adjustment set.
3.  **Estimation:** Applied **Propensity Score Matching** to create a synthetic control group.
4.  **Robustness Checks:**
    * *Placebo Test:* Replaced treatment with random noise $\rightarrow$ Effect dropped to ~0.0 (Valid).
    * *Random Common Cause:* Added noise confounder $\rightarrow$ Estimate remained stable (Robust).

### 💻 How to Run
1.  **Setup Environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Synthetic Data (Simulating Production Logs):**
    ```bash
    python src/data_loader.py
    ```
3.  **Run the Causal Engine:**
    ```bash
    python src/causal_pipeline.py
    ```

### 📈 Results Snapshot
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Naive Estimate** | $14.52 | Biased (Correlation != Causation) |
| **Causal Estimate** | **$9.98** | Corrected using DoWhy |
| **Ground Truth** | $10.00 | Validated via Data Generation Process |

---
*Author: [Your Name]*