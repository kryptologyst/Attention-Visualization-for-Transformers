"""Unit tests for attention visualization components."""

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from attention_viz.core import AttentionVisualizer, get_device, set_seed
from attention_viz.methods.advanced import AdvancedAttentionAnalyzer
from attention_viz.metrics.evaluation import AttentionEvaluationMetrics


class TestAttentionVisualizer:
    """Test cases for AttentionVisualizer class."""
    
    def test_device_detection(self):
        """Test device detection functionality."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_seed_setting(self):
        """Test seed setting functionality."""
        set_seed(42)
        # This is a basic test - in practice, you'd test reproducibility
        assert True  # Placeholder for actual reproducibility test
    
    def test_visualizer_initialization(self):
        """Test AttentionVisualizer initialization."""
        visualizer = AttentionVisualizer()
        assert visualizer.model_name == "bert-base-uncased"
        assert visualizer.device is not None
        assert visualizer.model is None
        assert visualizer.tokenizer is None
    
    def test_visualizer_custom_model(self):
        """Test AttentionVisualizer with custom model."""
        visualizer = AttentionVisualizer(model_name="distilbert-base-uncased")
        assert visualizer.model_name == "distilbert-base-uncased"
    
    @pytest.mark.slow
    def test_model_loading(self):
        """Test model loading functionality."""
        visualizer = AttentionVisualizer()
        tokenizer, model = visualizer.load_model()
        
        assert tokenizer is not None
        assert model is not None
        assert visualizer.model is not None
        assert visualizer.tokenizer is not None
    
    @pytest.mark.slow
    def test_attention_extraction(self):
        """Test attention weight extraction."""
        visualizer = AttentionVisualizer()
        visualizer.load_model()
        
        text = "Hello world"
        attentions, tokens = visualizer.get_attention_weights(text)
        
        assert isinstance(attentions, list)
        assert isinstance(tokens, list)
        assert len(attentions) > 0
        assert len(tokens) > 0
        
        # Check attention tensor shapes
        for layer_attn in attentions:
            assert isinstance(layer_attn, torch.Tensor)
            assert len(layer_attn.shape) == 4  # [batch, heads, seq, seq]


class TestAdvancedAttentionAnalyzer:
    """Test cases for AdvancedAttentionAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
        device = torch.device("cpu")
        return AdvancedAttentionAnalyzer(model, tokenizer, device)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.model is not None
        assert analyzer.tokenizer is not None
        assert analyzer.device is not None
    
    @pytest.mark.slow
    def test_attention_pattern_analysis(self, analyzer):
        """Test attention pattern analysis."""
        # Create dummy attention weights
        batch_size, num_heads, seq_len = 1, 6, 10
        attention_weights = [
            torch.rand(batch_size, num_heads, seq_len, seq_len)
            for _ in range(3)
        ]
        tokens = ["token"] * seq_len
        
        patterns = analyzer.analyze_attention_patterns(attention_weights, tokens)
        
        assert isinstance(patterns, dict)
        assert "avg_entropy" in patterns
        assert "avg_sparsity" in patterns
        assert "avg_symmetry" in patterns
    
    @pytest.mark.slow
    def test_attention_flow_computation(self, analyzer):
        """Test attention flow computation."""
        # Create dummy attention weights
        batch_size, num_heads, seq_len = 1, 6, 10
        attention_weights = [
            torch.rand(batch_size, num_heads, seq_len, seq_len)
            for _ in range(3)
        ]
        tokens = ["token"] * seq_len
        
        flow_matrix = analyzer.compute_attention_flow_matrix(attention_weights, tokens)
        
        assert isinstance(flow_matrix, torch.Tensor)
        assert flow_matrix.shape == (seq_len, seq_len)
    
    @pytest.mark.slow
    def test_head_importance_computation(self, analyzer):
        """Test head importance computation."""
        # Create dummy attention weights
        batch_size, num_heads, seq_len = 1, 6, 10
        attention_weights = [
            torch.rand(batch_size, num_heads, seq_len, seq_len)
            for _ in range(3)
        ]
        tokens = ["token"] * seq_len
        
        importance_scores = analyzer.compute_attention_head_importance(
            attention_weights, tokens
        )
        
        assert isinstance(importance_scores, list)
        assert len(importance_scores) == len(attention_weights)
        
        for scores in importance_scores:
            assert isinstance(scores, torch.Tensor)
            assert scores.size(0) == num_heads


class TestAttentionEvaluationMetrics:
    """Test cases for AttentionEvaluationMetrics class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = AttentionEvaluationMetrics()
        assert evaluator is not None
    
    def test_stability_metrics(self):
        """Test stability metrics computation."""
        evaluator = AttentionEvaluationMetrics()
        
        # Create dummy attention weights for multiple runs
        batch_size, num_heads, seq_len = 1, 6, 10
        attention_weights_list = [
            [
                torch.rand(batch_size, num_heads, seq_len, seq_len)
                for _ in range(3)
            ]
            for _ in range(2)
        ]
        
        stability = evaluator.compute_stability_metrics(attention_weights_list)
        
        assert isinstance(stability, dict)
        assert "stability_mean" in stability
        assert "stability_std" in stability
    
    def test_interpretability_metrics(self):
        """Test interpretability metrics computation."""
        evaluator = AttentionEvaluationMetrics()
        
        # Create dummy attention weights
        batch_size, num_heads, seq_len = 1, 6, 10
        attention_weights = [
            torch.rand(batch_size, num_heads, seq_len, seq_len)
            for _ in range(3)
        ]
        tokens = ["token"] * seq_len
        
        metrics = evaluator.compute_interpretability_metrics(attention_weights, tokens)
        
        assert isinstance(metrics, dict)
        assert "avg_entropy" in metrics
        assert "avg_sparsity" in metrics
        assert "avg_symmetry" in metrics
        assert "avg_concentration" in metrics


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Initialize visualizer
        visualizer = AttentionVisualizer()
        
        # Load model
        tokenizer, model = visualizer.load_model()
        
        # Create analyzer and evaluator
        analyzer = AdvancedAttentionAnalyzer(model, tokenizer, visualizer.device)
        evaluator = AttentionEvaluationMetrics()
        
        # Test text
        text = "The quick brown fox jumps over the lazy dog"
        
        # Get attention weights
        attentions, tokens = visualizer.get_attention_weights(text)
        
        # Analyze patterns
        patterns = analyzer.analyze_attention_patterns(attentions, tokens)
        
        # Compute interpretability metrics
        interpretability = evaluator.compute_interpretability_metrics(attentions, tokens)
        
        # Basic assertions
        assert len(attentions) > 0
        assert len(tokens) > 0
        assert isinstance(patterns, dict)
        assert isinstance(interpretability, dict)


if __name__ == "__main__":
    pytest.main([__file__])
