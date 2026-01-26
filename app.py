import streamlit as st
import pandas as pd
import joblib
import io
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.logger import logger
from src.engine import TaskDetector, get_model_from_registry
from src.pipeline import PipelineArchitect
from src.evaluation import ModelEvaluator, LeaderboardEngine

st.set_page_config(page_title="AutoFlowML", layout="wide")
st.title("🌊 AutoFlowML")

config = load_config(Path("config.yaml"))

# Sidebar Overrides
st.sidebar.header("⚙️ Global Settings")
config['settings']['cv_folds'] = st.sidebar.number_input("CV Folds", 2, 10, config['settings']['cv_folds'])
config['settings']['random_state'] = st.sidebar.number_input("Random Seed", 0, 9999, config['settings']['random_state'])

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    target_col = st.selectbox("Select Target", options=df.columns)
    
    if st.button("🚀 Run AutoML Leaderboard"):
        X, y = df.drop(columns=[target_col]), df[target_col]
        
        # 1. Detection
        detector = TaskDetector(target_column=target_col)
        task = detector.detect(y)
        st.info(f"Detected Task: **{task.upper()}**")

        # 2. Setup Architect & Evaluator
        evaluator = ModelEvaluator(task, config['evaluation'][task])
        architect = PipelineArchitect(config)
        leaderboard = LeaderboardEngine(config, task, evaluator)

        # 3. Competition
        with st.spinner("Training models and validating..."):
            df_res = leaderboard.run_competition(architect, X, y)
            st.session_state['leaderboard'] = df_res
            
            # 4. Finalize Winner
            winner_slug = df_res.iloc[0]['Model'].lower()
            model_cls = get_model_from_registry(task, winner_slug)
            
            # Retrain on full data for export
            final_pipe = architect.build_pipeline(model_cls())
            final_pipe.fit(X, y)
            st.session_state['final_pipeline'] = final_pipe
            st.session_state['winner_name'] = winner_slug

    # Display Results if they exist in state
    if 'leaderboard' in st.session_state:
        st.write("### 🏆 Leaderboard")
        st.dataframe(st.session_state['leaderboard'], use_container_width=True)
        
        # 5. Download Section
        st.write("### 💾 Export Winner")
        buffer = io.BytesIO()
        joblib.dump(st.session_state['final_pipeline'], buffer)
        st.download_button(
            label=f"Download {st.session_state['winner_name'].upper()} Pipeline (.pkl)",
            data=buffer.getvalue(),
            file_name=f"autoflow_{st.session_state['winner_name']}.pkl",
            mime="application/octet-stream"
        )