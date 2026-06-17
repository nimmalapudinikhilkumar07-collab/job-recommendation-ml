Job Recommendation System (Machine Learning)

Overview

This project predicts whether a user should be recommended for a job based on their skills and job requirements.

The project uses feature engineering techniques such as skill matching, match ratio, and skill difference, followed by machine learning models for prediction.

Features

- Skill matching between users and job requirements
- Feature engineering
- Logistic Regression
- Random Forest
- Hyperparameter tuning using RandomizedSearchCV
- Threshold-based prediction
- Feature importance analysis

Dataset Columns

- User_Skills
- Job_Requirements
- Match_Score
- Recommended

Engineered Features

- match_count
- total_job_skills
- total_user_skills
- match_ratio
- skill_diff

Models Used

1. Logistic Regression
2. Random Forest
3. Tuned Random Forest

Evaluation Metrics

- Accuracy
- Precision
- Recall
- ROC-AUC Score
- Confusion Matrix

Key Learning

A major observation was that the Match_Score feature strongly influenced the target variable (Recommended). Comparing models with and without Match_Score helped understand feature importance and potential data leakage issues.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

Author

Nikhil Kumar
