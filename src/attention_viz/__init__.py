"""Core attention visualization methods for transformer models.

This module provides modern, type-safe implementations of attention visualization
techniques for transformer models, with support for various visualization methods
and evaluation metrics.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        Available torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class AttentionVisualizer:
    """Main class for attention visualization in transformer models.
    
    This class provides methods to load transformer models, extract attention
    weights, and visualize them using various techniques.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the attention visualizer.
        
        Args:
            model_name: Name of the transformer model to load.
            device: Device to run the model on. If None, auto-detects.
            seed: Random seed for reproducibility.
        """
        set_seed(seed)
        self.model_name = model_name
        self.device = device or get_device()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
        logger.info(f"Initializing AttentionVisualizer with model: {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """Load the transformer model and tokenizer.
        
        Returns:
            Tuple of (tokenizer, model).
            
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with attention outputs enabled
            self.model = AutoModel.from_pretrained(
                self.model_name,
                output_attentions=True,
                torch_dtype=torch.float32,
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return self.tokenizer, self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def get_attention_weights(
        self,
        text: str,
        return_tokens: bool = True,
    ) -> Union[Tuple[List[torch.Tensor], List[str]], List[torch.Tensor]]:
        """Extract attention weights for the given text.
        
        Args:
            text: Input text to analyze.
            return_tokens: Whether to return tokenized tokens.
            
        Returns:
            Attention weights tensor or tuple of (weights, tokens).
            
        Raises:
            RuntimeError: If model is not loaded or processing fails.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract attention weights
            attentions = outputs.attentions  # List of tensors, one per layer
            
            # Get tokens for visualization
            if return_tokens:
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                return attentions, tokens
            
            return attentions
            
        except Exception as e:
            logger.error(f"Failed to get attention weights: {e}")
            raise RuntimeError(f"Attention extraction failed: {e}")
    
    def visualize_attention_heatmap(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        layer: int = 0,
        head: int = 0,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "viridis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create a heatmap visualization of attention weights.
        
        Args:
            attention: Attention weights tensor.
            tokens: List of token strings.
            layer: Layer index to visualize.
            head: Attention head index to visualize.
            figsize: Figure size tuple.
            cmap: Colormap for the heatmap.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Extract attention for specific layer and head
        attention_matrix = attention[layer][0, head].cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap=cmap,
            cbar=True,
            ax=ax,
        )
        
        ax.set_title(f"Attention Weights - Layer {layer + 1}, Head {head + 1}")
        ax.set_xlabel("Tokens Attended To")
        ax.set_ylabel("Tokens Attending")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Attention heatmap saved to: {save_path}")
        
        return fig
    
    def visualize_attention_flow(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        layer: int = 0,
        head: int = 0,
        threshold: float = 0.1,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create an attention flow visualization showing connections between tokens.
        
        Args:
            attention: Attention weights tensor.
            tokens: List of token strings.
            layer: Layer index to visualize.
            head: Attention head index to visualize.
            threshold: Minimum attention weight to show connection.
            figsize: Figure size tuple.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        attention_matrix = attention[layer][0, head].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a network-style visualization
        n_tokens = len(tokens)
        positions = np.linspace(0, 1, n_tokens)
        
        # Plot tokens
        for i, token in enumerate(tokens):
            ax.scatter(positions[i], 0, s=100, alpha=0.7)
            ax.text(positions[i], -0.05, token, ha="center", va="top", rotation=45)
        
        # Plot attention connections
        for i in range(n_tokens):
            for j in range(n_tokens):
                if attention_matrix[i, j] > threshold:
                    alpha = min(attention_matrix[i, j] * 2, 1.0)
                    ax.plot(
                        [positions[i], positions[j]],
                        [0, 0],
                        alpha=alpha,
                        linewidth=attention_matrix[i, j] * 5,
                        color="blue",
                    )
        
        ax.set_title(f"Attention Flow - Layer {layer + 1}, Head {head + 1}")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, 0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Attention flow saved to: {save_path}")
        
        return fig
    
    def compute_attention_rollout(
        self,
        attention: List[torch.Tensor],
        discard_ratio: float = 0.9,
    ) -> torch.Tensor:
        """Compute attention rollout across all layers.
        
        Args:
            attention: List of attention tensors from all layers.
            discard_ratio: Ratio of attention weights to discard.
            
        Returns:
            Rolled out attention matrix.
        """
        # Initialize with identity matrix
        rollout = torch.eye(attention[0].size(-1)).to(attention[0].device)
        
        for layer_attention in attention:
            # Average across heads
            layer_attention = layer_attention.mean(dim=1)  # [batch, seq, seq]
            
            # Apply rollout
            rollout = torch.matmul(layer_attention[0], rollout)
        
        return rollout
    
    def visualize_attention_rollout(
        self,
        attention: List[torch.Tensor],
        tokens: List[str],
        discard_ratio: float = 0.9,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize attention rollout across all layers.
        
        Args:
            attention: List of attention tensors from all layers.
            tokens: List of token strings.
            discard_ratio: Ratio of attention weights to discard.
            figsize: Figure size tuple.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        rollout = self.compute_attention_rollout(attention, discard_ratio)
        rollout_matrix = rollout.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            rollout_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            cbar=True,
            ax=ax,
        )
        
        ax.set_title("Attention Rollout (All Layers)")
        ax.set_xlabel("Tokens Attended To")
        ax.set_ylabel("Tokens Attending")
        
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Attention rollout saved to: {save_path}")
        
        return fig


def main() -> None:
    """Example usage of the AttentionVisualizer."""
    # Initialize visualizer
    visualizer = AttentionVisualizer()
    
    # Load model
    tokenizer, model = visualizer.load_model()
    
    # Example text
    text = "The quick brown fox jumps over the lazy dog"
    
    # Get attention weights
    attentions, tokens = visualizer.get_attention_weights(text)
    
    # Visualize attention for different layers and heads
    for layer in range(min(3, len(attentions))):
        for head in range(min(2, attentions[layer].size(1))):
            fig = visualizer.visualize_attention_heatmap(
                attentions, tokens, layer=layer, head=head
            )
            plt.show()
    
    # Visualize attention rollout
    fig = visualizer.visualize_attention_rollout(attentions, tokens)
    plt.show()


if __name__ == "__main__":
    main()
