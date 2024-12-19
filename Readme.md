# Results


## News Category Classification Project

This project implements various machine learning models to classify news articles into different categories based on their headlines.

### Implementation Details

The project uses the following classifiers:
- Custom Naive Bayes Implementation
- Support Vector Machine (SVM)
- Decision Tree
- Logistic Regression

### Data Processing
- Dataset: News Category Dataset v3 from HuffPost
- Features: Headlines converted to TF-IDF vectors (max 5000 features)
- Train-test split: 80-20 ratio

### Results Summary

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 46.94% |
| SVM | 55.70% |
| Decision Tree | 37.79% |
| Logistic Regression | 55.72% |

### Key Findings

1. **Best Performing Models**: 
   - SVM and Logistic Regression performed similarly, achieving ~56% accuracy
   - Both showed better balanced performance across categories

2. **Category-specific Performance**:
   - Strong performance in categories like:
     - Politics (F1: 0.73-0.75)
     - Weddings (F1: 0.74)
     - Style & Beauty (F1: ~0.70)
   - Challenging categories:
     - Arts (F1: 0.19-0.21)
     - U.S. News (F1: 0.07-0.09)

3. **Model Characteristics**:
   - Naive Bayes: Shows high precision but lower recall
   - SVM/Logistic Regression: More balanced precision-recall trade-off
   - Decision Tree: Lowest overall performance but simpler interpretability

### Code Structure

The implementation includes:
- Custom Naive Bayes classifier implementation
- Vectorization using TF-IDF
- Model training and evaluation pipeline
- Comprehensive performance metrics calculation
