import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import logger
from src.engine import TaskDetector, get_model_from_registry
from src.pipeline import PipelineArchitect
from src.evaluation import ModelEvaluator, LeaderboardEngine, CrossValidator
from src.visualizer import ModelVisualizer

from src.processing import TargetEncodedModelWrapper


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AutoFlowML", layout="wide", page_icon="🌊")
st.title("🌊 AutoFlowML: Professional AutoML Suite")

# 1. LOAD CONFIGURATION
config = load_config(Path("config.yaml"))

# --- SIDEBAR: REPRODUCIBILITY & THRESHOLDS ---
st.sidebar.header("⚙️ Global Settings")
# These sync directly with our Engine and Evaluation classes
config['settings']['cv_folds'] = st.sidebar.slider("CV Folds", 2, 10, config['settings']['cv_folds'])
config['settings']['random_state'] = st.sidebar.number_input("Random Seed (Global)", 0, 9999, config['settings']['random_state'])

st.sidebar.subheader("Cleaning Overrides")
config['cleaning']['nan_thresholds']['numeric'] = st.sidebar.slider("Numeric NaN Drop %", 0.0, 1.0, 0.5)
config['cleaning']['variance']['min_threshold'] = st.sidebar.number_input("Min Variance", 0.0, 1.0, 0.01, format="%.4f")

# 2. DATA INGESTION
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Data Preview", df.head())
    target_col = st.selectbox("Select Target Column", options=df.columns)
    
    if st.button("🚀 Run AutoML Competition"):
        # --- LOGGING AUDIT ---
        with st.expander("🛠️ Internal Processing Logs"):
            for handler in logger.handlers:
                if hasattr(handler, 'logs'):
                    for log in handler.logs:
                        st.code(log)
                    handler.logs = [] # Reset for next interaction
        # Reset session state for new runs
        for key in ['leaderboard', 'pipelines', 'oof_preds', 'task', 'y_true']:
            if key in st.session_state: del st.session_state[key]
            
        X, y = df.drop(columns=[target_col]), df[target_col]
        
        # --- ENGINE: TASK DETECTION ---
        detector = TaskDetector(target_column=target_col)
        task = detector.detect(y)
        st.session_state['task'] = task
        st.session_state['y_true'] = y
        st.info(f"Detected Task: **{task.upper()}**")

        # --- EVALUATION: COMPETITION ---
        evaluator = ModelEvaluator(task, config['evaluation'][task])
        architect = PipelineArchitect(config)
        leaderboard_engine = LeaderboardEngine(config, task, evaluator)

        with st.spinner("Executing Stratified Cross-Validation on all candidate models..."):
            # We need the LeaderboardEngine to return the OOF preds for all models now
            # To meet your request, we run the competition
            df_res = leaderboard_engine.run_competition(architect, X, y)
            st.session_state['leaderboard'] = df_res
            
            # Pre-calculate Pipelines & OOF data for ALL models for the 'Shopping Cart'
            st.session_state['pipelines'] = {}
            st.session_state['oof_data'] = {}
            
            # Secondary Loop: Final Training & OOF Storage
            # (In a V2, this would be inside LeaderboardEngine for efficiency)
            cv_engine = CrossValidator(config['settings']['cv_folds'], config['settings']['random_state'])
            
            for _, row in df_res.iterrows():
                m_slug = row['Model'].lower()
                m_cls = get_model_from_registry(task, m_slug)()
                
                # OOF Predictions for Visuals
                pipe_for_cv = architect.build_pipeline(m_cls, task_type=task)
                cv_res = cv_engine.run_cv(pipe_for_cv, X, y, task)
                st.session_state['oof_data'][m_slug] = cv_res['oof_predictions']
                
                # Final Fit for Download
                final_pipe = architect.build_pipeline(m_cls, task_type=task)
                final_pipe.fit(X, y)
                st.session_state['pipelines'][m_slug] = final_pipe

    # --- UI DISPLAY: LEADERBOARD & VISUALS ---
    if 'leaderboard' in st.session_state:
        st.divider()
        top_col, export_col = st.columns([0.7, 0.3])

        with top_col:
            st.write("### 🏆 Model Leaderboard")
            st.dataframe(
                st.session_state['leaderboard'].style.highlight_min(axis=0, subset=['CV Loss (Avg)'], color='#1e3d24'),
                use_container_width=True
            )

        with export_col:
            st.write("### 💾 Export & Select")
            selected_model = st.selectbox(
                "Pick a model to visualize/download:", 
                options=st.session_state['leaderboard']['Model'].tolist()
            )
            
            m_slug = selected_model.lower()
            
            # Download Logic
            buffer = io.BytesIO()
            joblib.dump(st.session_state['pipelines'][m_slug], buffer)
            st.download_button(
                label=f"📥 Download {selected_model} (.pkl)",
                data=buffer.getvalue(),
                file_name=f"autoflow_{m_slug}.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )

        # --- VISUALIZATION SECTION ---
        st.write(f"### 📈 {selected_model} Performance Insights")
        visualizer = ModelVisualizer()
        y_true = st.session_state['y_true']
        oof_preds = st.session_state['oof_data'][m_slug]
        task = st.session_state['task']

        viz_col1, viz_col2 = st.columns(2)

        if task == "classification":
            with viz_col1:
                labels = sorted(y_true.unique().tolist())
                fig_cm = visualizer.plot_confusion_matrix(y_true, oof_preds, labels=labels)
                st.plotly_chart(fig_cm, use_container_width=True)
            with viz_col2:
                st.write("**Model Note:** Classification heatmaps show the weighted average precision/recall across folds.")
        else:
            with viz_col1:
                fig_act = visualizer.plot_prediction_vs_actual(y_true, oof_preds)
                st.plotly_chart(fig_act, use_container_width=True)
            with viz_col2:
                fig_res = visualizer.plot_regression_residuals(y_true, oof_preds)
                st.plotly_chart(fig_res, use_container_width=True)

        # --- FEATURE IMPORTANCE SECTION ---
        # 1. Get the fitted pipeline for the selected model
        best_pipe = st.session_state['pipelines'][m_slug]
        
        # 2. Extract feature names after processing
        # Scikit-learn 1.2+ provides get_feature_names_out() for pipelines
        try:
            processed_features = best_pipe.named_steps['processing'].get_feature_names_out()
            
            # 3. Get importances from the wrapped model (thanks to our proxy!)
            model_step = best_pipe.named_steps['model']
            
            # Some models use feature_importances_, others use coef_
            if hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
            elif hasattr(model_step, 'coef_'):
                importances = np.abs(model_step.coef_).flatten()
            else:
                importances = None

            if importances is not None:
                st.write("#### 🔑 Feature Importance")
                fig_importance = visualizer.plot_feature_importance(importances, processed_features)
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Feature importance is not available for this model type.")
                
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            st.info("Feature importance visualization is only available for supported models.")