#!/usr/bin/env python3
"""Example script demonstrating attention visualization functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from attention_viz.core import AttentionVisualizer
from attention_viz.methods.advanced import AdvancedAttentionAnalyzer
from attention_viz.metrics.evaluation import AttentionEvaluationMetrics


def main():
    """Run attention visualization example."""
    print("🧠 Attention Visualization Example")
    print("=" * 50)
    
    # Initialize visualizer
    print("Initializing visualizer...")
    visualizer = AttentionVisualizer(model_name="distilbert-base-uncased")
    
    # Load model
    print("Loading model...")
    tokenizer, model = visualizer.load_model()
    print(f"✓ Model loaded: {visualizer.model_name}")
    print(f"✓ Device: {visualizer.device}")
    
    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming the world",
        "Attention mechanisms are powerful tools for understanding",
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n📝 Example {i}: {text}")
        print("-" * 30)
        
        # Get attention weights
        attentions, tokens = visualizer.get_attention_weights(text)
        print(f"✓ Tokens: {tokens}")
        print(f"✓ Layers: {len(attentions)}")
        print(f"✓ Heads per layer: {attentions[0].size(1)}")
        
        # Advanced analysis
        analyzer = AdvancedAttentionAnalyzer(model, tokenizer, visualizer.device)
        patterns = analyzer.analyze_attention_patterns(attentions, tokens)
        
        print("📊 Attention Patterns:")
        for key, value in patterns.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluation metrics
        evaluator = AttentionEvaluationMetrics()
        interpretability = evaluator.compute_interpretability_metrics(attentions, tokens)
        
        print("📈 Interpretability Metrics:")
        for key, value in interpretability.items():
            print(f"  {key}: {value:.4f}")
    
    print("\n🎉 Example completed successfully!")
    print("\nTo run the interactive demo:")
    print("  streamlit run demo/app.py")
    print("\nTo run tests:")
    print("  pytest tests/ -v")


if __name__ == "__main__":
    main()
