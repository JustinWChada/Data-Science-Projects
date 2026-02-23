import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as stats

dataframe = pd.read_csv('ai_impact_student_performance_dataset.csv')

#analyzing relationships between AI usage and student performance
# print(dataframe.info())

study_time = dataframe["ai_usage_time_minutes"]
final_score = dataframe["final_score"]

correlation, pvalue = stats.pearsonr(study_time,final_score)
# print(f"Correlation between AI Usage Time and Final Exam Scores: {correlation}, P-value: {pvalue}")
plt.scatter(study_time, final_score, color = 'blue', alpha=0.5)
plt.xlabel("AI Use Hours Per Day")
plt.ylabel("Final Exam Scores")
plt.title("Final Exam Scores vs Study Hours Per Day")
plt.grid(True)
plt.show()

#spearman correlation

spearman_corr, pvalue = stats.spearmanr(study_time, final_score)
print(f"Spearman Correlation between AI Use Hours Per Day and Final Exam Scores: {spearman_corr}, P-value: {pvalue}")

"""
Spearman Correlation between AI Use Hours Per Day and Final Exam Scores: -0.0027718886334318965, P-value: 0.8042218451527122
- This indicates no meaningful monotonic relationship between AI usage hours and exam scores.
- The negative sign suggests a very slight inverse trend (more AI use → slightly lower scores), but the effect is so tiny it’s essentially negligible
- The p-value (0.804) is much greater than the common threshold of 0.05.
- This means the observed correlation is not statistically significant — it could easily be due to random chance

- AI use hours per day do not appear to influence exam scores in this dataset.
- Students who used AI more did not consistently perform better or worse than those who used it less.
- Other factors (like study hours, prior knowledge, or exam difficulty) are likely more important predictors of exam performance

"""

# print(dataframe.columns)

#AI Usage time vs Final Exam Scores

dataframe = dataframe.dropna(subset=['ai_usage_time_minutes', 'final_score'])

correlation = dataframe['ai_usage_time_minutes'].corr(dataframe['final_score'], method = 'pearson')
print("Pearson Corr: {}".format(correlation))

correlation = dataframe['ai_usage_time_minutes'].corr(dataframe['final_score'], method = 'spearman')
print("Spearman Corr: {}".format(correlation))

"""
AI usage time has no meaningful relationship with exam scores in this dataset
"""

# Extract the data
x = dataframe["ai_usage_time_minutes"]
y = dataframe["final_score"]

# Create the scatter plot
plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.7, label="Data points")

# Fit a regression line using SciPy
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope * x + intercept
plt.plot(x, line, color="red", label="Regression line")

# Add labels and title
plt.title("Correlation between AI Study Time and Academic Performance")
plt.xlabel("AI Study Time")
plt.ylabel("Academic Performance")
plt.legend()
plt.show()

"""
Neutral effect: AI study time, at least in this dataset, does not appear to influence academic performance. Students who spent more time with AI tools did not perform better or worse overall.
Flat regression line: The red line is essentially horizontal, which means there is no detectable trend between AI study time and academic performance.
Scatter distribution: The blue data points are spread without any clear upward or downward pattern. This reinforces the idea that increased AI study time does not consistently lead to higher or lower scores.
Reference Img: Correlation - AI Study Time vs Final Score.png
"""

#-  does AI Usage significantly affect performance distribution.

# Chi-square test: Does AI usage significantly affect performance category distribution?
contingency_table = pd.crosstab(dataframe['uses_ai'], dataframe['performance_category'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("Chi-square test result:", chi2, "p-value:", p)

"""
The chi-square test reinforces the earlier findings: AI usage is statistically independent of exam performance categories. Both continuous (time spent) and categorical (use vs. non-use) analyses point to the same conclusion — AI usage is not a predictor of academic success here.
"""