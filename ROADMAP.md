# 🌊 AutoFlowML: Project Roadmap & Evolution

## 📍 Phase 1: The Robust Core (Completed) ✅

*Based on commits `aa4ff68` through `1fb97aa*`

* **Infrastructure & DevOps**
* ✅ **CI/CD Pipeline**: Integrated GitHub Actions for automated testing and workflow validation (`e99dc8c`, `816eb48`).
* ✅ **Logging Architecture**: Implemented a dual-handler system for both Console and Streamlit UI (`3fe3178`).
* ✅ **Modular Testing**: Comprehensive unit test suite covering `config`, `preprocessors`, `engines`, and `cleaning` (`43638d7`, `b3f330e`, `a33726a`).


* **Data & Model Engineering**
* ✅ **Cleaning Guardrails**: Developed custom transformers, including the **Cardinality Stripper** and **Universal Dropper** (`1fb97aa`, `2567841`).
* ✅ **Transparent Proxy Model**: Engineered a `TargetEncodedModelWrapper` to handle classification label decoding natively within the pipeline (`1fb97aa`).
* ✅ **Automated Pipeline**: End-to-end pipeline assembly using dynamic feature selection (`f04161e`).


* **UX & Evaluation**
* ✅ **Leaderboard Engine**: Multi-model benchmarking with custom evaluation metrics (`ead0470`, `9261a74`).
* ✅ **Interactive Visuals**: Plotly-based model diagnostics (Confusion Matrix, Residuals) integrated into Streamlit (`1fb97aa`).



---

## 🏗️ Phase 2: MLOps & Experiment Tracking (Current) 🧪

*Goal: Moving from a local tool to a trackable ML platform.*

* **📊 Experiment Management (MLflow)**
* 🕒 **Tracking Server**: Log every AutoML "Run" (hyperparams, metrics, and tags) to MLflow.
* 🕒 **Model Artifacts**: Save the entire fitted Scikit-Learn pipeline as an MLflow Model flavor.


* **🛠️ Architectural Hardening**
* 🕒 **Type Hinting (PEP 484)**: Apply strict typing across `src/` to improve IDE support and catch bugs early.
* 🕒 **Pydantic Config**: Migrate `config.yaml` loading to Pydantic models for strict runtime validation.



---

## 🕒 Phase 3: Performance & Explainability (Planned) 💎

*Goal: Optimization and "Glass-Box" transparency.*

* **🎯 Advanced Optimization**
* 🕒 **Regression Target Clipping**: Enhancing the regression tasks performance by clipping the target.
* 🕒 **Adding Robust Scaling Handlers**: enhancing the pipeline by implementing techniques implemented in `AgriYield` project.
* 🕒 **Hyperparameter Tuning**: Integrate **Optuna** for Bayesian optimization of the leaderboard winner.
* 🕒 **Model Explainability**: Add **SHAP** integration to explain individual predictions (local) and feature impacts (global).


* **🔌 Deployment Ready**
* 🕒 **REST API Generation**: Automated **FastAPI** script generation for serving the winning pipeline.
* 🕒 **Containerization**: Create multi-stage `Dockerfiles` for lightweight model deployment.



---

### 💡 What you should do next (The "Pro" Sequence):

1. **Add Type Hints**: Go through `src/engine/detector.py` and `src/pipeline/architect.py` and add type hints. It’s a low-effort, high-visibility task that makes your code look "Senior."
2. **Integrate MLflow**:
* In your `LeaderboardEngine.run_competition`, wrap the model training in an `with mlflow.start_run(run_name=slug):` block.
* Log the `CV Loss` and the `duration`.


3. **Refactor Config with Pydantic**:
* This is a strong "Software Engineering" signal. Instead of `config['cleaning']['variance']`, you’d use `config.cleaning.variance`. It prevents typos from breaking your code.