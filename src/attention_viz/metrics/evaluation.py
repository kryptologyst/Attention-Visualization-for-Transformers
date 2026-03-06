"""Evaluation metrics for attention visualization quality.

This module provides metrics to evaluate the quality and faithfulness of
attention visualizations, including stability, faithfulness, and interpretability
metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class AttentionEvaluationMetrics:
    """Evaluation metrics for attention visualization quality."""
    
    def __init__(self) -> None:
        """Initialize the evaluation metrics."""
        pass
    
    def compute_stability_metrics(
        self,
        attention_weights_list: List[List[torch.Tensor]],
        method: str = "kendall_tau",
    ) -> Dict[str, float]:
        """Compute stability metrics across multiple runs.
        
        Args:
            attention_weights_list: List of attention weights from multiple runs.
            method: Stability metric method ('kendall_tau', 'spearman', 'iou').
            
        Returns:
            Dictionary of stability metrics.
        """
        if len(attention_weights_list) < 2:
            return {"stability": 0.0}
        
        stability_scores = []
        
        # Compare each pair of runs
        for i in range(len(attention_weights_list)):
            for j in range(i + 1, len(attention_weights_list)):
                attn_i = attention_weights_list[i]
                attn_j = attention_weights_list[j]
                
                # Compute stability for each layer
                layer_stabilities = []
                for layer_idx in range(min(len(attn_i), len(attn_j))):
                    layer_stability = self._compute_layer_stability(
                        attn_i[layer_idx], attn_j[layer_idx], method
                    )
                    layer_stabilities.append(layer_stability)
                
                stability_scores.append(np.mean(layer_stabilities))
        
        return {
            "stability_mean": np.mean(stability_scores),
            "stability_std": np.std(stability_scores),
            "stability_min": np.min(stability_scores),
            "stability_max": np.max(stability_scores),
        }
    
    def _compute_layer_stability(
        self,
        attn_i: torch.Tensor,
        attn_j: torch.Tensor,
        method: str,
    ) -> float:
        """Compute stability between two attention matrices.
        
        Args:
            attn_i: First attention matrix.
            attn_j: Second attention matrix.
            method: Stability metric method.
            
        Returns:
            Stability score.
        """
        # Average across heads and batch
        attn_i_avg = attn_i.mean(dim=(0, 1)).flatten().cpu().numpy()
        attn_j_avg = attn_j.mean(dim=(0, 1)).flatten().cpu().numpy()
        
        if method == "kendall_tau":
            tau, _ = kendalltau(attn_i_avg, attn_j_avg)
            return abs(tau) if not np.isnan(tau) else 0.0
        elif method == "spearman":
            rho, _ = spearmanr(attn_i_avg, attn_j_avg)
            return abs(rho) if not np.isnan(rho) else 0.0
        elif method == "iou":
            # Convert to binary masks
            threshold = 0.1
            mask_i = (attn_i_avg > threshold).astype(float)
            mask_j = (attn_j_avg > threshold).astype(float)
            
            intersection = np.sum(mask_i * mask_j)
            union = np.sum(np.maximum(mask_i, mask_j))
            
            return intersection / union if union > 0 else 0.0
        else:
            raise ValueError(f"Unknown stability method: {method}")
    
    def compute_faithfulness_metrics(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        target_token_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute faithfulness metrics for attention weights.
        
        Args:
            model: The transformer model.
            tokenizer: Tokenizer for the model.
            text: Input text.
            attention_weights: Attention weights to evaluate.
            tokens: List of token strings.
            target_token_idx: Index of target token for evaluation.
            
        Returns:
            Dictionary of faithfulness metrics.
        """
        faithfulness_scores = {}
        
        # Deletion test: Remove tokens with high attention and measure performance drop
        deletion_score = self._compute_deletion_score(
            model, tokenizer, text, attention_weights, tokens, target_token_idx
        )
        faithfulness_scores["deletion_score"] = deletion_score
        
        # Insertion test: Add tokens with high attention and measure performance gain
        insertion_score = self._compute_insertion_score(
            model, tokenizer, text, attention_weights, tokens, target_token_idx
        )
        faithfulness_scores["insertion_score"] = insertion_score
        
        # Sufficiency test: Use only top-attended tokens
        sufficiency_score = self._compute_sufficiency_score(
            model, tokenizer, text, attention_weights, tokens, target_token_idx
        )
        faithfulness_scores["sufficiency_score"] = sufficiency_score
        
        return faithfulness_scores
    
    def _compute_deletion_score(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        target_token_idx: Optional[int],
    ) -> float:
        """Compute deletion score for faithfulness evaluation."""
        try:
            # Get baseline prediction
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            baseline_logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state.mean()
            
            # Compute average attention across layers
            avg_attention = torch.zeros(len(tokens))
            for layer_attn in attention_weights:
                layer_avg = layer_attn.mean(dim=(0, 1))  # Average across heads and batch
                avg_attention += layer_avg.mean(dim=0)  # Average across source tokens
            
            avg_attention /= len(attention_weights)
            
            # Sort tokens by attention
            attention_scores = avg_attention.cpu().numpy()
            sorted_indices = np.argsort(attention_scores)[::-1]
            
            # Remove top-attended tokens and measure performance drop
            performance_drops = []
            for k in range(1, min(len(tokens) // 2, 10)):
                # Remove top k tokens
                tokens_to_remove = sorted_indices[:k]
                modified_text = " ".join([
                    token for i, token in enumerate(tokens) 
                    if i not in tokens_to_remove
                ])
                
                if len(modified_text.strip()) == 0:
                    continue
                
                # Get prediction on modified text
                modified_inputs = tokenizer(modified_text, return_tensors="pt")
                with torch.no_grad():
                    modified_outputs = model(**modified_inputs)
                modified_logits = modified_outputs.logits if hasattr(modified_outputs, "logits") else modified_outputs.last_hidden_state.mean()
                
                # Compute performance drop
                if target_token_idx is not None:
                    drop = baseline_logits[0, target_token_idx] - modified_logits[0, target_token_idx]
                else:
                    drop = torch.norm(baseline_logits - modified_logits)
                
                performance_drops.append(drop.item())
            
            # Compute area under the deletion curve
            if performance_drops:
                return np.mean(performance_drops)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Deletion score computation failed: {e}")
            return 0.0
    
    def _compute_insertion_score(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        target_token_idx: Optional[int],
    ) -> float:
        """Compute insertion score for faithfulness evaluation."""
        try:
            # Get baseline prediction
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            baseline_logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state.mean()
            
            # Compute average attention across layers
            avg_attention = torch.zeros(len(tokens))
            for layer_attn in attention_weights:
                layer_avg = layer_attn.mean(dim=(0, 1))
                avg_attention += layer_avg.mean(dim=0)
            
            avg_attention /= len(attention_weights)
            
            # Sort tokens by attention
            attention_scores = avg_attention.cpu().numpy()
            sorted_indices = np.argsort(attention_scores)[::-1]
            
            # Add top-attended tokens and measure performance gain
            performance_gains = []
            for k in range(1, min(len(tokens) // 2, 10)):
                # Add top k tokens
                tokens_to_add = sorted_indices[:k]
                added_text = " ".join([
                    tokens[i] for i in tokens_to_add
                ])
                modified_text = f"{added_text} {text}"
                
                # Get prediction on modified text
                modified_inputs = tokenizer(modified_text, return_tensors="pt")
                with torch.no_grad():
                    modified_outputs = model(**modified_inputs)
                modified_logits = modified_outputs.logits if hasattr(modified_outputs, "logits") else modified_outputs.last_hidden_state.mean()
                
                # Compute performance gain
                if target_token_idx is not None:
                    gain = modified_logits[0, target_token_idx] - baseline_logits[0, target_token_idx]
                else:
                    gain = torch.norm(modified_logits - baseline_logits)
                
                performance_gains.append(gain.item())
            
            # Compute area under the insertion curve
            if performance_gains:
                return np.mean(performance_gains)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Insertion score computation failed: {e}")
            return 0.0
    
    def _compute_sufficiency_score(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        target_token_idx: Optional[int],
    ) -> float:
        """Compute sufficiency score for faithfulness evaluation."""
        try:
            # Get baseline prediction
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            baseline_logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state.mean()
            
            # Compute average attention across layers
            avg_attention = torch.zeros(len(tokens))
            for layer_attn in attention_weights:
                layer_avg = layer_attn.mean(dim=(0, 1))
                avg_attention += layer_avg.mean(dim=0)
            
            avg_attention /= len(attention_weights)
            
            # Sort tokens by attention
            attention_scores = avg_attention.cpu().numpy()
            sorted_indices = np.argsort(attention_scores)[::-1]
            
            # Use only top-attended tokens and measure performance
            sufficiency_scores = []
            for k in range(1, min(len(tokens), 10)):
                # Use only top k tokens
                tokens_to_use = sorted_indices[:k]
                modified_text = " ".join([
                    tokens[i] for i in tokens_to_use
                ])
                
                if len(modified_text.strip()) == 0:
                    continue
                
                # Get prediction on modified text
                modified_inputs = tokenizer(modified_text, return_tensors="pt")
                with torch.no_grad():
                    modified_outputs = model(**modified_inputs)
                modified_logits = modified_outputs.logits if hasattr(modified_outputs, "logits") else modified_outputs.last_hidden_state.mean()
                
                # Compute sufficiency score
                if target_token_idx is not None:
                    sufficiency = modified_logits[0, target_token_idx] / baseline_logits[0, target_token_idx]
                else:
                    sufficiency = torch.norm(modified_logits) / torch.norm(baseline_logits)
                
                sufficiency_scores.append(sufficiency.item())
            
            # Compute area under the sufficiency curve
            if sufficiency_scores:
                return np.mean(sufficiency_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Sufficiency score computation failed: {e}")
            return 0.0
    
    def compute_interpretability_metrics(
        self,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
    ) -> Dict[str, float]:
        """Compute interpretability metrics for attention weights.
        
        Args:
            attention_weights: List of attention tensors from all layers.
            tokens: List of token strings.
            
        Returns:
            Dictionary of interpretability metrics.
        """
        metrics = {}
        
        # Compute attention entropy (diversity)
        entropies = []
        for layer_attn in attention_weights:
            avg_attn = layer_attn.mean(dim=(0, 1))  # Average across heads and batch
            entropy = -torch.sum(avg_attn * torch.log(avg_attn + 1e-8), dim=-1)
            entropies.append(entropy.mean().item())
        
        metrics["avg_entropy"] = np.mean(entropies)
        metrics["entropy_std"] = np.std(entropies)
        
        # Compute attention sparsity (focus)
        sparsities = []
        for layer_attn in attention_weights:
            avg_attn = layer_attn.mean(dim=(0, 1))
            sparsity = (avg_attn > 0.01).float().mean().item()
            sparsities.append(sparsity)
        
        metrics["avg_sparsity"] = np.mean(sparsities)
        metrics["sparsity_std"] = np.std(sparsities)
        
        # Compute attention symmetry
        symmetries = []
        for layer_attn in attention_weights:
            avg_attn = layer_attn.mean(dim=(0, 1))
            symmetry = torch.norm(avg_attn - avg_attn.T).item()
            symmetries.append(symmetry)
        
        metrics["avg_symmetry"] = np.mean(symmetries)
        metrics["symmetry_std"] = np.std(symmetries)
        
        # Compute attention concentration
        concentrations = []
        for layer_attn in attention_weights:
            avg_attn = layer_attn.mean(dim=(0, 1))
            max_attn = torch.max(avg_attn, dim=-1)[0]
            concentrations.append(max_attn.mean().item())
        
        metrics["avg_concentration"] = np.mean(concentrations)
        metrics["concentration_std"] = np.std(concentrations)
        
        return metrics
    
    def compute_comprehensive_evaluation(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        target_token_idx: Optional[int] = None,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute comprehensive evaluation metrics.
        
        Args:
            model: The transformer model.
            tokenizer: Tokenizer for the model.
            text: Input text.
            attention_weights: Attention weights to evaluate.
            tokens: List of token strings.
            target_token_idx: Index of target token for evaluation.
            
        Returns:
            Dictionary of comprehensive evaluation metrics.
        """
        evaluation = {}
        
        # Faithfulness metrics
        faithfulness = self.compute_faithfulness_metrics(
            model, tokenizer, text, attention_weights, tokens, target_token_idx
        )
        evaluation["faithfulness"] = faithfulness
        
        # Interpretability metrics
        interpretability = self.compute_interpretability_metrics(attention_weights, tokens)
        evaluation["interpretability"] = interpretability
        
        # Overall score (weighted combination)
        overall_score = (
            0.4 * faithfulness["deletion_score"] +
            0.3 * faithfulness["insertion_score"] +
            0.2 * faithfulness["sufficiency_score"] +
            0.1 * interpretability["avg_entropy"]
        )
        evaluation["overall_score"] = overall_score
        
        return evaluation
