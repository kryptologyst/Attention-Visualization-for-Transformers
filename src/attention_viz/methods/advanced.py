"""Advanced attention visualization methods.

This module provides advanced attention visualization techniques including
gradient-based attention, attention flow analysis, and multi-head attention
aggregation methods.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency
from captum.attr._utils.attribution import Attribution
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class AdvancedAttentionAnalyzer:
    """Advanced attention analysis methods for transformer models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ) -> None:
        """Initialize the advanced attention analyzer.
        
        Args:
            model: Pre-trained transformer model.
            tokenizer: Tokenizer for the model.
            device: Device to run computations on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_gradient_attention(
        self,
        text: str,
        target_token_idx: Optional[int] = None,
        method: str = "integrated_gradients",
    ) -> torch.Tensor:
        """Compute gradient-based attention weights.
        
        Args:
            text: Input text to analyze.
            target_token_idx: Index of target token for attribution.
            method: Attribution method ('integrated_gradients' or 'saliency').
            
        Returns:
            Gradient-based attention weights.
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get input embeddings
        input_ids = inputs["input_ids"]
        embeddings = self.model.get_input_embeddings()(input_ids)
        
        # Define forward function for attribution
        def forward_func(embeds: torch.Tensor) -> torch.Tensor:
            # Replace input embeddings
            inputs_copy = inputs.copy()
            inputs_copy["inputs_embeds"] = embeds
            if "input_ids" in inputs_copy:
                del inputs_copy["input_ids"]
            
            outputs = self.model(**inputs_copy)
            return outputs.last_hidden_state
        
        # Compute attribution
        if method == "integrated_gradients":
            attributor = IntegratedGradients(forward_func)
            attributions = attributor.attribute(
                embeddings,
                target=target_token_idx,
                n_steps=50,
            )
        elif method == "saliency":
            attributor = Saliency(forward_func)
            attributions = attributor.attribute(embeddings, target=target_token_idx)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Convert to attention-like format
        attention_weights = torch.norm(attributions, dim=-1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        return attention_weights
    
    def analyze_attention_patterns(
        self,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
    ) -> Dict[str, float]:
        """Analyze attention patterns and compute statistics.
        
        Args:
            attention_weights: List of attention tensors from all layers.
            tokens: List of token strings.
            
        Returns:
            Dictionary of attention pattern statistics.
        """
        stats = {}
        
        # Compute attention entropy (diversity of attention)
        entropies = []
        for layer_attn in attention_weights:
            # Average across heads and batch
            avg_attn = layer_attn.mean(dim=(0, 1))  # [seq, seq]
            
            # Compute entropy for each token
            entropy = -torch.sum(avg_attn * torch.log(avg_attn + 1e-8), dim=-1)
            entropies.append(entropy.mean().item())
        
        stats["avg_entropy"] = np.mean(entropies)
        stats["entropy_std"] = np.std(entropies)
        
        # Compute attention sparsity (how focused attention is)
        sparsities = []
        for layer_attn in attention_weights:
            avg_attn = layer_attn.mean(dim=(0, 1))
            # Count non-zero attention weights
            sparsity = (avg_attn > 0.01).float().mean().item()
            sparsities.append(sparsity)
        
        stats["avg_sparsity"] = np.mean(sparsities)
        stats["sparsity_std"] = np.std(sparsities)
        
        # Compute attention symmetry (how symmetric attention matrix is)
        symmetries = []
        for layer_attn in attention_weights:
            avg_attn = layer_attn.mean(dim=(0, 1))
            # Compute symmetry score
            symmetry = torch.norm(avg_attn - avg_attn.T).item()
            symmetries.append(symmetry)
        
        stats["avg_symmetry"] = np.mean(symmetries)
        stats["symmetry_std"] = np.std(symmetries)
        
        return stats
    
    def compute_attention_flow_matrix(
        self,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
    ) -> torch.Tensor:
        """Compute attention flow matrix across all layers.
        
        Args:
            attention_weights: List of attention tensors from all layers.
            tokens: List of token strings.
            
        Returns:
            Attention flow matrix.
        """
        # Initialize flow matrix
        seq_len = len(tokens)
        flow_matrix = torch.eye(seq_len).to(self.device)
        
        # Accumulate attention flow through layers
        for layer_attn in attention_weights:
            # Average across heads and batch
            avg_attn = layer_attn.mean(dim=(0, 1))  # [seq, seq]
            
            # Update flow matrix
            flow_matrix = torch.matmul(avg_attn, flow_matrix)
        
        return flow_matrix
    
    def visualize_attention_flow_matrix(
        self,
        flow_matrix: torch.Tensor,
        tokens: List[str],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize the attention flow matrix.
        
        Args:
            flow_matrix: Attention flow matrix.
            tokens: List of token strings.
            figsize: Figure size tuple.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        flow_np = flow_matrix.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(flow_np, cmap="viridis", aspect="auto")
        
        # Set labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        ax.set_title("Attention Flow Matrix (All Layers)")
        ax.set_xlabel("Target Tokens")
        ax.set_ylabel("Source Tokens")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Attention flow matrix saved to: {save_path}")
        
        return fig
    
    def compute_attention_head_importance(
        self,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
    ) -> List[torch.Tensor]:
        """Compute importance scores for each attention head.
        
        Args:
            attention_weights: List of attention tensors from all layers.
            tokens: List of token strings.
            
        Returns:
            List of importance scores for each layer.
        """
        importance_scores = []
        
        for layer_idx, layer_attn in enumerate(attention_weights):
            # Compute importance for each head
            head_importance = []
            
            for head_idx in range(layer_attn.size(1)):
                head_attn = layer_attn[0, head_idx]  # [seq, seq]
                
                # Compute attention diversity (entropy)
                entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-8), dim=-1)
                avg_entropy = entropy.mean()
                
                # Compute attention concentration (max attention)
                max_attn = torch.max(head_attn, dim=-1)[0]
                avg_max_attn = max_attn.mean()
                
                # Combine metrics
                importance = avg_entropy * avg_max_attn
                head_importance.append(importance.item())
            
            importance_scores.append(torch.tensor(head_importance))
        
        return importance_scores
    
    def visualize_head_importance(
        self,
        importance_scores: List[torch.Tensor],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize attention head importance across layers.
        
        Args:
            importance_scores: List of importance scores for each layer.
            figsize: Figure size tuple.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap of head importance
        max_heads = max(scores.size(0) for scores in importance_scores)
        importance_matrix = np.zeros((len(importance_scores), max_heads))
        
        for layer_idx, scores in enumerate(importance_scores):
            importance_matrix[layer_idx, :scores.size(0)] = scores.numpy()
        
        im = ax.imshow(importance_matrix, cmap="viridis", aspect="auto")
        
        # Set labels
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Layer")
        ax.set_title("Attention Head Importance")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Head importance visualization saved to: {save_path}")
        
        return fig
    
    def compute_attention_similarity(
        self,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
    ) -> torch.Tensor:
        """Compute similarity between attention patterns across layers.
        
        Args:
            attention_weights: List of attention tensors from all layers.
            tokens: List of token strings.
            
        Returns:
            Similarity matrix between layers.
        """
        n_layers = len(attention_weights)
        similarity_matrix = torch.zeros(n_layers, n_layers)
        
        for i in range(n_layers):
            for j in range(n_layers):
                # Average attention across heads and batch
                attn_i = attention_weights[i].mean(dim=(0, 1))
                attn_j = attention_weights[j].mean(dim=(0, 1))
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    attn_i.flatten(), attn_j.flatten(), dim=0
                )
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def visualize_layer_similarity(
        self,
        similarity_matrix: torch.Tensor,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize similarity between attention patterns across layers.
        
        Args:
            similarity_matrix: Similarity matrix between layers.
            figsize: Figure size tuple.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        similarity_np = similarity_matrix.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(similarity_np, cmap="coolwarm", vmin=-1, vmax=1)
        
        # Set labels
        n_layers = similarity_matrix.size(0)
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        ax.set_xticklabels([f"Layer {i+1}" for i in range(n_layers)])
        ax.set_yticklabels([f"Layer {i+1}" for i in range(n_layers)])
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        ax.set_title("Attention Pattern Similarity Between Layers")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Layer similarity visualization saved to: {save_path}")
        
        return fig
