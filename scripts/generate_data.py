"""Generate synthetic data for attention visualization experiments."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer


def generate_synthetic_texts(n_samples: int = 100) -> List[str]:
    """Generate synthetic text samples for attention analysis.
    
    Args:
        n_samples: Number of text samples to generate.
        
    Returns:
        List of synthetic text samples.
    """
    # Template sentences with different structures
    templates = [
        "The {adjective} {noun} {verb} {preposition} the {adjective2} {noun2}",
        "{noun} is {adjective} and {verb} {adverb}",
        "When {noun} {verb}, the {noun2} becomes {adjective}",
        "The {adjective} {noun} {verb} because it is {adjective2}",
        "{noun} and {noun2} are both {adjective}",
    ]
    
    # Word banks
    adjectives = ["quick", "brown", "lazy", "smart", "fast", "slow", "bright", "dark", "happy", "sad"]
    nouns = ["fox", "dog", "cat", "bird", "fish", "tree", "house", "car", "book", "computer"]
    verbs = ["jumps", "runs", "flies", "swims", "reads", "writes", "thinks", "learns", "teaches", "helps"]
    prepositions = ["over", "under", "through", "around", "between", "above", "below", "near", "far", "inside"]
    adverbs = ["quickly", "slowly", "carefully", "easily", "hardly", "barely", "almost", "nearly", "quite", "very"]
    
    texts = []
    for _ in range(n_samples):
        template = random.choice(templates)
        
        # Fill template with random words
        text = template.format(
            adjective=random.choice(adjectives),
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            preposition=random.choice(prepositions),
            adjective2=random.choice(adjectives),
            noun2=random.choice(nouns),
            adverb=random.choice(adverbs),
        )
        
        texts.append(text)
    
    return texts


def create_dataset_metadata(texts: List[str]) -> Dict[str, Any]:
    """Create metadata for the synthetic dataset.
    
    Args:
        texts: List of text samples.
        
    Returns:
        Dataset metadata dictionary.
    """
    # Analyze text characteristics
    word_counts = [len(text.split()) for text in texts]
    
    metadata = {
        "name": "synthetic_attention_dataset",
        "description": "Synthetic text dataset for attention visualization experiments",
        "n_samples": len(texts),
        "avg_length": np.mean(word_counts),
        "min_length": min(word_counts),
        "max_length": max(word_counts),
        "std_length": np.std(word_counts),
        "features": {
            "text": {
                "type": "string",
                "description": "Input text for attention analysis",
                "max_length": 512,
            }
        },
        "sensitive_attributes": [],
        "monotonicity_constraints": {},
        "generated_by": "synthetic_data_generator",
        "version": "1.0.0",
    }
    
    return metadata


def save_dataset(texts: List[str], metadata: Dict[str, Any], output_dir: Path) -> None:
    """Save the synthetic dataset to files.
    
    Args:
        texts: List of text samples.
        metadata: Dataset metadata.
        output_dir: Output directory path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save texts
    with open(output_dir / "texts.json", "w") as f:
        json.dump(texts, f, indent=2)
    
    # Save metadata
    with open(output_dir / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save as text file for easy reading
    with open(output_dir / "texts.txt", "w") as f:
        for i, text in enumerate(texts, 1):
            f.write(f"{i:3d}: {text}\n")
    
    print(f"✓ Dataset saved to: {output_dir}")
    print(f"✓ {len(texts)} samples generated")
    print(f"✓ Average length: {metadata['avg_length']:.1f} words")


def main():
    """Generate synthetic dataset."""
    print("🔧 Generating Synthetic Dataset")
    print("=" * 40)
    
    # Generate texts
    n_samples = 100
    print(f"Generating {n_samples} synthetic text samples...")
    texts = generate_synthetic_texts(n_samples)
    
    # Create metadata
    metadata = create_dataset_metadata(texts)
    
    # Save dataset
    output_dir = Path("data/synthetic")
    save_dataset(texts, metadata, output_dir)
    
    # Show sample texts
    print("\n📝 Sample texts:")
    for i, text in enumerate(texts[:5], 1):
        print(f"  {i}: {text}")
    
    print(f"\n🎉 Dataset generation completed!")
    print(f"📁 Files created:")
    print(f"  - {output_dir}/texts.json")
    print(f"  - {output_dir}/meta.json")
    print(f"  - {output_dir}/texts.txt")


if __name__ == "__main__":
    main()
