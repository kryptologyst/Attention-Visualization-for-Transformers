"""Streamlit demo application for attention visualization.

This module provides an interactive web interface for exploring attention
visualization in transformer models.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from attention_viz.core import AttentionVisualizer
from attention_viz.methods.advanced import AdvancedAttentionAnalyzer
from attention_viz.metrics.evaluation import AttentionEvaluationMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Attention Visualization Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
DISCLAIMER = """
**DISCLAIMER**: This attention visualization tool is for research and educational purposes only. 
Attention weights may not always reflect the true reasoning process of the model and should not 
be used as the sole basis for critical decisions. Always combine with human judgment and 
domain expertise.
"""

def load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    """Load the transformer model and tokenizer."""
    try:
        with st.spinner(f"Loading {model_name}..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True,
                torch_dtype=torch.float32,
            )
            model.eval()
            return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def main() -> None:
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Attention Visualization Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f'<div class="warning-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = {
        "BERT Base": "bert-base-uncased",
        "BERT Large": "bert-large-uncased",
        "DistilBERT": "distilbert-base-uncased",
        "RoBERTa Base": "roberta-base",
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0,
    )
    model_name = model_options[selected_model]
    
    # Load model
    if "model" not in st.session_state or st.session_state.get("current_model") != model_name:
        tokenizer, model = load_model(model_name)
        if tokenizer is not None and model is not None:
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.current_model = model_name
            st.session_state.visualizer = AttentionVisualizer(model_name)
            st.session_state.analyzer = AdvancedAttentionAnalyzer(model, tokenizer, torch.device("cpu"))
            st.session_state.evaluator = AttentionEvaluationMetrics()
        else:
            st.error("Failed to load model. Please try again.")
            return
    
    # Input text
    st.sidebar.header("Input")
    default_text = "The quick brown fox jumps over the lazy dog"
    input_text = st.sidebar.text_area(
        "Enter text to analyze",
        value=default_text,
        height=100,
    )
    
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
        return
    
    # Visualization options
    st.sidebar.header("Visualization Options")
    
    viz_method = st.sidebar.selectbox(
        "Visualization Method",
        ["Heatmap", "Flow", "Rollout", "Head Importance", "Layer Similarity"],
        index=0,
    )
    
    # Layer and head selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        max_layers = min(12, st.session_state.model.config.num_hidden_layers)
        selected_layer = st.selectbox("Layer", range(max_layers), index=0)
    
    with col2:
        max_heads = min(12, st.session_state.model.config.num_attention_heads)
        selected_head = st.selectbox("Head", range(max_heads), index=0)
    
    # Analysis options
    st.sidebar.header("Analysis")
    show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=True)
    show_patterns = st.sidebar.checkbox("Show Attention Patterns", value=False)
    
    # Main content
    if st.sidebar.button("Analyze Attention", type="primary"):
        try:
            # Get attention weights
            with st.spinner("Computing attention weights..."):
                attentions, tokens = st.session_state.visualizer.get_attention_weights(input_text)
            
            # Display tokens
            st.subheader("Tokenized Input")
            st.write("Tokens:", tokens)
            
            # Create visualizations
            if viz_method == "Heatmap":
                fig = st.session_state.visualizer.visualize_attention_heatmap(
                    attentions, tokens, layer=selected_layer, head=selected_head
                )
                st.pyplot(fig)
                
            elif viz_method == "Flow":
                fig = st.session_state.visualizer.visualize_attention_flow(
                    attentions, tokens, layer=selected_layer, head=selected_head
                )
                st.pyplot(fig)
                
            elif viz_method == "Rollout":
                fig = st.session_state.visualizer.visualize_attention_rollout(
                    attentions, tokens
                )
                st.pyplot(fig)
                
            elif viz_method == "Head Importance":
                importance_scores = st.session_state.analyzer.compute_attention_head_importance(
                    attentions, tokens
                )
                fig = st.session_state.analyzer.visualize_head_importance(importance_scores)
                st.pyplot(fig)
                
            elif viz_method == "Layer Similarity":
                similarity_matrix = st.session_state.analyzer.compute_attention_similarity(
                    attentions, tokens
                )
                fig = st.session_state.analyzer.visualize_layer_similarity(similarity_matrix)
                st.pyplot(fig)
            
            # Show evaluation metrics
            if show_metrics:
                st.subheader("Evaluation Metrics")
                
                with st.spinner("Computing evaluation metrics..."):
                    evaluation = st.session_state.evaluator.compute_comprehensive_evaluation(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        input_text,
                        attentions,
                        tokens,
                    )
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Overall Score", f"{evaluation['overall_score']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Deletion Score", f"{evaluation['faithfulness']['deletion_score']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Insertion Score", f"{evaluation['faithfulness']['insertion_score']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed metrics
                with st.expander("Detailed Metrics"):
                    st.json(evaluation)
            
            # Show attention patterns
            if show_patterns:
                st.subheader("Attention Pattern Analysis")
                
                patterns = st.session_state.analyzer.analyze_attention_patterns(
                    attentions, tokens
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Entropy", f"{patterns['avg_entropy']:.3f}")
                with col2:
                    st.metric("Avg Sparsity", f"{patterns['avg_sparsity']:.3f}")
                with col3:
                    st.metric("Avg Symmetry", f"{patterns['avg_symmetry']:.3f}")
                with col4:
                    st.metric("Entropy Std", f"{patterns['entropy_std']:.3f}")
        
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            logger.error(f"Analysis error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note**: This demo is for educational purposes. Attention weights may not always "
        "reflect the true reasoning process of the model."
    )

if __name__ == "__main__":
    main()
