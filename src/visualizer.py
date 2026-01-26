import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from typing import List, Dict

class ModelVisualizer:
    """Generates interactive plots for model evaluation."""

    def __init__(self):
        pass

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, labels: List[str]) -> go.Figure:
        """
        Generates a Plotly heatmap for a confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(x="Predicted Label", y="True Label", color="Count")
        )
        fig.update_xaxes(side="bottom")
        fig.update_layout(
            title_text='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        return fig

    def plot_regression_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
        """
        Generates a Plotly scatter plot of residuals vs. predictions.
        """
        residuals = y_true - y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(opacity=0.6),
            name='Residuals'
        ))
        fig.add_trace(go.Scatter(
            x=[min(y_pred), max(y_pred)],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Zero Residual Line'
        ))
        fig.update_layout(
            title_text='Residuals vs. Predictions',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals (True - Predicted)',
            hovermode="closest"
        )
        return fig
    
    def plot_prediction_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
        """
        Generates a Plotly scatter plot of actual vs. predicted values for regression.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker=dict(opacity=0.6),
            name='Predictions'
        ))
        fig.add_trace(go.Scatter(
            x=[min(y_true), max(y_true)],
            y=[min(y_true), max(y_true)],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Ideal Prediction Line'
        ))
        fig.update_layout(
            title_text='Actual vs. Predicted Values',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            hovermode="closest"
        )
        return fig
    def plot_feature_importance(self, importances: np.ndarray, feature_names: List[str], top_n: int = 10) -> go.Figure:
        """
        Generates a horizontal bar chart of the top N features by importance.
        """
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(top_n)

        fig = px.bar(
            df_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances',
            color='Importance',
            color_continuous_scale='Blugrn'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        return fig