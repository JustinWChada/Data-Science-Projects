import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stats
from scipy.stats import zscore
from scipy.stats import mode

df = pd.read_csv("ai_impact_student_performance_dataset.csv")

ai_cols = [
    "uses_ai",
    "ai_usage_time_minutes",
    "ai_tools_used",
    "ai_usage_purpose", 
    "ai_dependency_score",
    "ai_generated_content_percentage",
    "ai_prompts_per_week"
]

ai_df = df[ai_cols].copy()
# print(ai_df.head())
ai_df = ai_df.dropna()
# print(ai_df.shape)
# print(ai_df.head())

numeric_features = [
    "ai_usage_time_minutes",
    "ai_dependency_score",
    "ai_generated_content_percentage",
    "ai_prompts_per_week",
]

categorical_features = [
    "uses_ai",
    "ai_usage_purpose",
    "ai_tools_used",   # moved here
]



from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# numeric and categorical feature lists
numeric_features = [
    "ai_usage_time_minutes",
    "ai_dependency_score",
    "ai_generated_content_percentage",
    "ai_prompts_per_week",
]

categorical_features = [
    "uses_ai",
    "ai_usage_purpose",
    "ai_tools_used",
]

# select data and drop missing
ai_df = df[numeric_features + categorical_features].dropna()

# scale numeric
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(ai_df[numeric_features])

# encode categoricals
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
categorical_encoded = ohe.fit_transform(ai_df[categorical_features])

# combine
import numpy as np
preprocessor = np.hstack([numeric_scaled, categorical_encoded])

# run k-means (define n_clusters as you want)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(preprocessor)

df.loc[ai_df.index, "ai_cluster"] = clusters


df.plot.scatter(
    x = "ai_usage_time_minutes",
    y = "ai_dependency_score",
    c = "ai_cluster",
    colormap = "viridis"
)
plt.title("Clusters by AI Usage time and dependency")
plt.show()


"""
The graph suggests that AI dependency is not simply a function of usage time. 
Instead, students fall into distinct clusters of behavior, which could be useful for tailoring interventions 
e.g., supporting highly dependent users differently than strategic, low-dependency users

"""