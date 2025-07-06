
# Credit Risk Scoring Service

This repository provides a complete solution for building, training, and deploying a credit risk prediction model tailored for **Bati Bank**. It includes tools for data processing, model development, API deployment, and experiment tracking.

---
# Credit Scoring Business Understanding

## Influence of Basel II Accord on Model Interpretability

The Basel II Accord emphasizes the importance of risk measurement for financial institutions. This focus necessitates models that are not only accurate but also interpretable and well-documented. Regulators and stakeholders require transparency to understand how risk assessments are made, which helps ensure compliance with regulatory standards. An interpretable model allows institutions to explain their credit decisions, reducing the potential for misunderstandings and promoting trust among clients and regulators.

## Necessity of Creating a Proxy Variable

In credit risk modeling, a direct "default" label may be unavailable, particularly in datasets where defaults are rare. Creating a proxy variable—such as a combination of behavioral indicators—becomes essential to estimate default risk. However, relying on proxy variables introduces potential business risks; if these proxies do not accurately reflect true default risk, they may lead to incorrect credit decisions, resulting in financial losses, reputational damage, and compliance issues.

## Trade-offs Between Simple and Complex Models

When choosing between a simple, interpretable model like Logistic Regression (using Weight of Evidence) and a complex model like Gradient Boosting, several trade-offs arise. 

- **Simplicity and Interpretability**: Logistic Regression offers straightforward interpretation, which is crucial in a regulated environment where stakeholders must understand risk decisions. This transparency can facilitate compliance with regulations and enhance trust.

- **Performance and Accuracy**: Gradient Boosting, while more complex, often provides better predictive performance. However, its "black box" nature may hinder interpretability, making it difficult for institutions to explain decisions to regulators and clients.

In a regulated context, the choice between these models often hinges on the balance between the need for transparency and the desire for accuracy. Institutions must weigh the risks of using complex models against the benefits of potentially improved predictive power.

## 🚀 Quickstart

### Build and Run the API

```bash
# Build the Docker image
docker compose build

# Start the API locally with hot-reloading
docker compose up
```

### Run Tests

```bash
pytest -q
```

---

## 🧠 Model Training & Tracking

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Launch MLflow UI

```bash
mlflow ui  # open in browser: http://127.0.0.1:5000
```

### Train and Evaluate Models

```bash
python -m src.train \
    --raw-path data/data.csv \
    --model-out artifacts/best_model.pkl
```

This script performs grid search and logs all runs to MLflow. The best model (based on AUC) is registered under `credit-risk-best`.

---

## 🧪 Development Guidelines

- Use feature branches and pull requests (PRs) targeting `main`.
- All new code must include unit tests and pass both `pytest` and `ruff` checks.
- Keep notebooks clean and concise. Move reusable code to the `src/` directory.
- Do not commit sensitive files (e.g., trained models, large datasets). Use external storage like DVC or S3.

---

## 💼 Business Context

### 1. Basel II Compliance & Interpretability

To comply with the **Basel II Internal Ratings-Based (IRB)** approach, credit models must be transparent, auditable, and regularly validated. Our design prioritizes:

- Clear feature definitions
- Monotonic scorecards
- Explainable models (e.g., logistic regression with WOE)

This ensures supervisors and auditors can trace each prediction back to its inputs.

### 2. Using a Proxy Target

As the dataset lacks a true default flag, a **proxy label** (e.g., risky customer segments or fraud flags) is created to train the model. This enables supervised learning but introduces label risk. Ongoing validation against actual defaults is crucial to avoid misclassification and mitigate regulatory and financial exposure.

### 3. Model Choices: Simple vs. Complex

| Model Type | Pros | Cons |
|------------|------|------|
| **Logistic Regression + WOE** | Highly interpretable, stress-test friendly | May sacrifice predictive accuracy |
| **Gradient Boosting** | Higher accuracy, non-linear patterns | Opaque, requires SHAP/PDPs for explanation |

**Recommendation:** Use a transparent model for production and compliance. Use more complex models for monitoring and experimentation.

---

## 📜 License

This project is licensed under the **MIT License**.

