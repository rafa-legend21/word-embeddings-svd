# Learning and Evaluating Word Embeddings with SVD

## Overview
This project builds word embeddings from a large-scale text corpus using a
co-occurrence matrix and truncated Singular Value Decomposition (SVD).
The embeddings are evaluated using similarity queries, analogy tasks,
and semantic analysis to assess their usefulness for downstream NLP applications.

## Data
The dataset consists of a word co-occurrence matrix derived from a Wikipedia corpus,
containing the **top 10,000 most frequent words**.
Co-occurrence counts are weighted by word distance and log-scaled to
address the power-law distribution of word frequencies.

## Methodology
- Log-scaling of co-occurrence matrix
- Truncated SVD for dimensionality reduction
- Row-normalized word embeddings
- Evaluation via cosine similarity and analogy tasks

## Results
- Embedding dimension: k = 300 
- Analogy task accuracy: 0.606983
- Similarity queries retrieve semantically related words
- Higher embedding dimensions improve analogy accuracy up to a point

## Applications
These embeddings can support NLP tasks such as semantic search,
document similarity, clustering, and feature extraction for downstream models.

## Project Structure
notebooks/
└── word_embeddings_svd.Rmd

## Notes
This project emphasizes interpretability and evaluation rather than
neural network–based embedding models.

