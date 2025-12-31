# Interpretable Word Embeddings and Model Risk Analysis

## Overview
This project examines word embeddings learned via linear decomposition methods,
with a focus on interpretability, stability, and bias as a form of model risk.
The analysis highlights how latent semantic dimensions can encode structured
associations present in training data.

## Motivation
In regulated or high-stakes decision environments, model transparency and
risk awareness are critical. This project demonstrates how even simple,
interpretable models can encode unintended biases if not carefully evaluated.

## Evaluation
- Stability across embedding dimensions
- Analogy task performance
- Semantic direction analysis
- Bias examination using gender-related word projections

## Key Observations
- Embedding performance stabilizes around **k = 300**
- Semantic directions reveal systematic associations
- Bias represents a form of model risk rather than noise

## Implications
These findings underscore the importance of validation, interpretability,
and bias awareness when embedding-based models are used in decision systems.
