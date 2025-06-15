# Stroke Prediction Project

## About The Project

This project aims to predict the likelihood of stroke using various machine learning models based on a provided healthcare dataset. The process involves data loading, cleaning, exploratory data analysis (EDA), feature engineering, and training and evaluating several classification algorithms.

## Built With

*   Python
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   tensorflow
*   imblearn

## Getting Started

### Dependencies

*   Python 3.6+
*   Required libraries are listed in the "Built With" section and can be installed using pip.

### Installation

1.  Clone the repository (if hosted on a platform like GitHub).
2.  Install the required libraries:
   bash pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imblearn
3.  Ensure the `healthcare-dataset-stroke-data.csv` file is in the same directory as the notebook.

## Usage

1.  Open the Jupyter Notebook or Google Colab file.
2.  Run the cells sequentially to perform data loading, preprocessing, EDA, model training, and evaluation.
3.  Observe the output of each cell, including visualizations, model performance metrics (accuracy, classification report, confusion matrix), and predictions.

## Roadmap

*   Explore hyperparameter tuning for the machine learning models.
*   Investigate other advanced classification algorithms.
*   Implement cross-validation for more robust model evaluation.
*   Consider feature selection techniques.

## Results

| Model             | Accuracy (Before SMOTE) | F1-Score (Class '1') (Before SMOTE) | Accuracy (After SMOTE) | F1-Score (Class '1') (After SMOTE) |
|-------------------|-------------------------|-------------------------------------|------------------------|------------------------------------|
| KNN               | ~0.97                   | ~0.00                               | ~0.97                  | ~0.97                              |
| Naive Bayes       | ~0.16                   | ~0.04                               | ~0.76                  | ~0.77                              |
| Logistic Regression| ~0.97                   | ~0.00                               | ~0.74                  | ~0.73                              |
| Decision Tree     | ~0.95                   | ~0.26                               | ~0.99                  | ~0.99                              |
| Random Forest     | ~0.96                   | ~0.00                               | ~0.99                  | ~0.99                              |
| SVM               | ~0.97                   | ~0.00                               | ~0.71                  | ~0.72                              |
| Neural Network    | ~0.97                   | ~0.00                               | ~0.99                  | ~0.99                              |

## Authors

Salma Elatries

## Acknowledgements

*   Dataset source: [Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)
*   Libraries and tools used: pandas, scikit-learn, tensorflow, etc.
