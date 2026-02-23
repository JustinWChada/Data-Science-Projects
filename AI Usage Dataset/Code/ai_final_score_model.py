import pandas as pd 
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("AI Usage Dataset/ai_impact_student_performance_dataset.csv")

# Define features and target variable
df = df.dropna()
# print(df.columns)
X = df[[
    "study_hours_per_day",
    "improvement_rate",
    "assignment_scores_avg",
    "last_exam_score",
    "ai_usage_purpose",
    #added
    "attendance_percentage",
    "concept_understanding_score",
    "study_consistency_index",
    "class_participation_score",
    "uses_ai",
    "ai_usage_time_minutes",
    "ai_dependency_score",
    "ai_generated_content_percentage",
    "ai_prompts_per_week",
    "ai_ethics_score",
    #lifestyle
    "sleep_hours",
    "social_media_hours",
    "tutoring_hours"
    #demographics
    #,"age", "gender", "grade_level", "ai_tools_used"
]]

y = df["final_score"]

X = pd.get_dummies(X, columns=['ai_usage_purpose'], drop_first=True) #,'gender','grade_level', 'ai_tools_used'

# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
) 

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

# Explanation:
# The output shows the shape of the X_train and X_test datasets. The first number in the tuple is the number of samples and the second number is the number of features. In this case, X_train has 144 samples and 5 features, while X_test has 36 samples and 5 features.


ln_model = LinearRegression()
ln_model.fit(X_train, y_train)

predictions = ln_model.predict(X_test)

print("Linear R² (Training):", r2_score(y_train, ln_model.predict(X_train)))
print("Linear R² (Testing):", r2_score(y_test, predictions))
print("Linear MSE:", mean_squared_error(y_test, predictions))

#NonLinear Regression - RandomForest

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("Random Forest R² (Train): ", r2_score(y_train, rf_model.predict(X_train)))
print("Random Forest R² (Test): ", r2_score(y_test, rf_predictions))
print("Random Forest MSE: ", mean_squared_error(y_test, rf_predictions))

"""
- Adding study behavior, AI usage, and lifestyle features made a huge difference — your model now explains ~85% of variance, which is excellent.
- Linear regression is surprisingly outperforming Random Forest in test accuracy, which suggests the relationships in your data are fairly linear once the right features are included.
- Random Forest could still be tuned (e.g., limiting depth, adjusting min_samples_leaf) to reduce overfitting and possibly close the gap

"""