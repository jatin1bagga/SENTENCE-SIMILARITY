# Sentence Embedding Model Evaluation using STS Benchmark and TOPSIS

## Overview
This project evaluates sentence embedding models using the **Semantic Textual Similarity Benchmark (STS-B)** dataset. The evaluation includes computing cosine similarity between sentence pairs, calculating Spearman correlation with human-annotated scores, and ranking models using the **TOPSIS** multi-criteria decision-making method.

## Features
- Evaluates multiple sentence embedding models from **Sentence-Transformers**.
- Computes cosine similarity between sentence pairs.
- Measures correlation with human-labeled similarity scores using **Spearman's Rank Correlation**.
- Measures model efficiency in terms of **execution time** and **model size**.
- Uses **TOPSIS** to rank models based on accuracy, execution time, and model size.
- Generates **visualizations** for decision-making.

## Dataset
The dataset used for evaluation is **STSB Multi-MT (English, Development Set)**, available from the Hugging Face `datasets` library.

## Requirements
Ensure the following Python packages are installed:

```bash
pip install sentence-transformers datasets scipy numpy pandas matplotlib seaborn
```

## Implementation Steps
1. **Load the Dataset**: Extracts 100 sentence pairs and their similarity scores.
2. **Evaluate Sentence Embedding Models**:
   - Encode sentence pairs using each model.
   - Compute cosine similarity scores.
   - Calculate **Spearman Correlation** with human-labeled scores.
   - Measure **execution time per sentence pair**.
   - Compute **model size in MB**.
3. **Rank Models using TOPSIS**:
   - Normalize the evaluation metrics.
   - Assign weights (**Accuracy: 50%, Execution Time: 30%, Model Size: 20%**).
   - Compute distance to ideal and negative ideal solutions.
   - Calculate the final **TOPSIS Score** and rank models.
4. **Generate Results and Visualizations**:
   - Saves normalized decision matrix (`normalized_matrix.csv`).
   - Saves TOPSIS results (`topsis_results.csv`).
   - Generates a heatmap for the normalized decision matrix.
   - Generates a bar chart of TOPSIS scores.

## Results
The results include:
- **Normalized Decision Matrix** (saved as `normalized_matrix.csv`).
- **TOPSIS Scores and Model Ranking** (saved as `topsis_results.csv`).
- **Heatmap of Normalized Metrics** (`normalized_matrix.png`).
- **Bar chart of TOPSIS Scores** (`topsis_scores.png`).

## Example Output
```plaintext
Normalized Decision Matrix:
                            Spearman Correlation  Execution Time (ms)  Model Size (MB)
paraphrase-MiniLM-L6-v2                   0.933                32.277            82.5611
all-MiniLM-L12-v2                          0.939                64.380           128.192

TOPSIS Results:
                     Model  Spearman Correlation  Execution Time (ms)  Model Size (MB)  TOPSIS Score  Rank
0  paraphrase-MiniLM-L6-v2                  0.933                32.277            82.5611       0.5813     2
1     all-MiniLM-L12-v2                     0.939                64.380           128.192       0.6724     1
```

## Visualization
- **Heatmap of Normalized Decision Matrix**
  ![Normalized Matrix Heatmap](normalized_matrix.png)
- **Bar Chart of TOPSIS Scores**
  ![TOPSIS Scores](topsis_scores.png)

## Conclusion
This project provides a structured evaluation of sentence embedding models by balancing accuracy, speed, and size. The **TOPSIS** method helps in selecting the best model based on multi-criteria decision-making.

## References
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

