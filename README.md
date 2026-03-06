# Attention Visualization for Transformers

A comprehensive toolkit for visualizing and analyzing attention mechanisms in transformer models. This project provides state-of-the-art attention visualization techniques with robust evaluation metrics and an interactive demo interface.

## ⚠️ DISCLAIMER

**IMPORTANT**: This tool is designed for research and educational purposes only. Attention visualizations may not always accurately reflect the true reasoning process of transformer models. These visualizations should not be used as the sole basis for critical decisions in regulated environments without human review and domain expertise.

## Features

- **Multiple Visualization Methods**: Heatmaps, attention flow, rollout, head importance, and layer similarity
- **Advanced Analysis**: Gradient-based attention, attention pattern analysis, and flow computation
- **Comprehensive Evaluation**: Faithfulness metrics, stability analysis, and interpretability measures
- **Interactive Demo**: Streamlit-based web interface for real-time exploration
- **Modern Architecture**: Type-safe, well-documented code with proper error handling
- **Device Support**: Automatic device detection (CUDA → MPS → CPU)

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Attention-Visualization-for-Transformers.git
cd Attention-Visualization-for-Transformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from attention_viz.core import AttentionVisualizer

# Initialize visualizer
visualizer = AttentionVisualizer()

# Load model
tokenizer, model = visualizer.load_model()

# Analyze text
text = "The quick brown fox jumps over the lazy dog"
attentions, tokens = visualizer.get_attention_weights(text)

# Visualize attention
fig = visualizer.visualize_attention_heatmap(
    attentions, tokens, layer=0, head=0
)
```

### Advanced Analysis

```python
from attention_viz.methods.advanced import AdvancedAttentionAnalyzer
from attention_viz.metrics.evaluation import AttentionEvaluationMetrics

# Create analyzer
analyzer = AdvancedAttentionAnalyzer(model, tokenizer, device)

# Analyze attention patterns
patterns = analyzer.analyze_attention_patterns(attentions, tokens)

# Compute evaluation metrics
evaluator = AttentionEvaluationMetrics()
evaluation = evaluator.compute_comprehensive_evaluation(
    model, tokenizer, text, attentions, tokens
)
```

## Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- Model selection (BERT, RoBERTa, DistilBERT)
- Interactive visualization controls
- Real-time evaluation metrics
- Multiple visualization methods

## Configuration

The project uses YAML configuration files. See `configs/default.yaml` for available options:

```yaml
model:
  name: "bert-base-uncased"
  device: "auto"
  seed: 42

visualization:
  figsize: [12, 10]
  cmap: "viridis"
  dpi: 300

evaluation:
  stability_method: "kendall_tau"
  faithfulness_tests: ["deletion", "insertion", "sufficiency"]
```

## Visualization Methods

### 1. Attention Heatmaps
Traditional attention weight visualization showing which tokens attend to which other tokens.

### 2. Attention Flow
Network-style visualization showing attention connections between tokens.

### 3. Attention Rollout
Aggregated attention across all layers showing the complete attention flow.

### 4. Head Importance
Analysis of which attention heads are most important for the model's predictions.

### 5. Layer Similarity
Comparison of attention patterns across different layers.

## Evaluation Metrics

### Faithfulness Metrics
- **Deletion Score**: Performance drop when removing high-attention tokens
- **Insertion Score**: Performance gain when adding high-attention tokens
- **Sufficiency Score**: Performance using only top-attended tokens

### Stability Metrics
- **Kendall's τ**: Rank correlation across multiple runs
- **Spearman's ρ**: Monotonic relationship strength
- **IoU**: Intersection over union of attention patterns

### Interpretability Metrics
- **Entropy**: Diversity of attention distribution
- **Sparsity**: Focus of attention patterns
- **Symmetry**: Symmetry of attention matrices
- **Concentration**: Peak attention strength

## Project Structure

```
attention-viz/
├── src/attention_viz/
│   ├── __init__.py
│   ├── core.py                 # Core visualization functionality
│   ├── methods/
│   │   └── advanced.py        # Advanced analysis methods
│   ├── metrics/
│   │   └── evaluation.py      # Evaluation metrics
│   └── utils/
│       └── config.py          # Configuration utilities
├── demo/
│   └── app.py                 # Streamlit demo application
├── configs/
│   └── default.yaml           # Default configuration
├── tests/
│   └── test_attention_viz.py  # Unit tests
├── data/                      # Data directory
├── assets/                    # Generated visualizations
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                  # This file
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=attention_viz --cov-report=html
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for quality checks

Install pre-commit hooks:

```bash
pre-commit install
```

### Type Checking

```bash
mypy src/
```

### Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

## Limitations and Considerations

1. **Model Dependency**: Results depend on the specific model architecture and training data
2. **Attention vs. Reasoning**: Attention weights may not always reflect true reasoning processes
3. **Computational Cost**: Some analysis methods require significant computational resources
4. **Interpretation**: Attention patterns require domain expertise for proper interpretation
5. **Stability**: Attention patterns may vary across different runs and model versions

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{attention_viz,
  title={Attention Visualization for Transformers},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Attention-Visualization-for-Transformers}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Captum for gradient-based attribution methods
- Streamlit for the interactive demo interface
- The broader XAI research community

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

---

**Remember**: This tool is for research and educational purposes. Always combine attention visualizations with human judgment and domain expertise for critical applications.
# Attention-Visualization-for-Transformers
