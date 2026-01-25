import streamlit as st
import pandas as pd
from src.utils.config_loader import load_config
from src.utils.logger import logger
from src.engine import TaskDetector, get_model_from_registry
from src.pipeline import PipelineArchitect
from pathlib import Path

# Setup Page
st.set_page_config(page_title="AutoFlowML", layout="wide")
st.title("🌊 AutoFlowML: Professional Pipeline Engine")

# 1. Load Configuration
config = load_config(Path("config.yaml"))

# 2. File Upload
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # 3. Target Selection
    target_col = st.selectbox("Select Target Column", options=df.columns)
    
    if st.button("🚀 Analyze & Build Pipeline"):
        # Setup UI Containers
        log_container = st.expander("🛠️ Processing Logs", expanded=True)
        
        # --- EXECUTION ENGINE ---
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Step 1: Detect Task
        detector = TaskDetector(target_column=target_col)
        detected_task = detector.detect(y)
        st.info(f"Detected Task: **{detected_task.upper()}**")

        # Step 2: Assemble Pipeline
        # (For now, we grab the first model in the config registry)
        model_name = config['model_selection']['models'][detected_task][0]
        model_class = get_model_from_registry(detected_task, model_name)
        
        # Split features for the architect
        num_features = X.select_dtypes(include=['number']).columns.tolist()
        cat_features = X.select_dtypes(exclude=['number']).columns.tolist()

        architect = PipelineArchitect(config)
        pipeline = architect.build_pipeline(model_class(), num_features, cat_features)

        # Step 3: Fit
        with st.spinner("Training pipeline..."):
            pipeline.fit(X, y)
        
        st.success("Pipeline built and trained successfully!")

        # --- LOG DISPLAY ---
        # Fetch logs from our custom handler and show them in the expander
        for handler in logger.handlers:
            if hasattr(handler, 'logs'):
                for log in handler.logs:
                    log_container.code(log)