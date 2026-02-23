import pandas as pd 
import scipy.stats as stats 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv("ai_impact_student_performance_dataset.csv")

grouped_df = df.groupby("ai_usage_purpose").agg({
    'improvement_rate': 'mean',
    'last_exam_score': 'mean',
    'assignment_scores_avg': 'mean',
    'final_score': 'mean'
}).reset_index()

# print(grouped_df.sort_values('improvement_rate', ascending=False))
# print(grouped_df.head(20))

# create a figure with 4 subplots
# fig is the figure object
# axes is a list of 4 AxesSubplot objects
# figsize is a tuple of (width, height) in inches
# sharey is a boolean indicating whether the subplots share the y-axis
fig, axes = plt.subplots(1,4, figsize = (12,6), sharey = False)

grouped_df.plot(
    x = 'ai_usage_purpose',
    y = 'final_score',
    kind = 'bar',
    ax = axes[0],
    legend = False
)
axes[0].set_title("Final Score by AI Usage Purpose")
axes[0].set_ylabel("Final Score")
axes[0].set_xlabel("AI Usage Purpose")

grouped_df.plot(
    x = 'ai_usage_purpose',
    y = 'improvement_rate',
    kind = 'bar',
    ax =axes[1],
    legend = False
)

axes[1].set_title("Improvement Rate by AI Usage Purpose")
axes[1].set_ylabel("Improvement Rate")
axes[1].set_xlabel("AI Usage Purpose")

# third plot
grouped_df.plot(
    x = "ai_usage_purpose",
    y = 'assignment_scores_avg',
    kind = 'bar',
    ax = axes[2],
    legend = False
)
axes[2].set_title("Assignment Scores by AI Usage Purpose")
axes[2].set_ylabel("Assignment Scores")
axes[2].set_xlabel("AI Usage Purpose")

# fourth plot
grouped_df.plot(
    x = 'ai_usage_purpose',
    y = 'last_exam_score',
    kind = 'bar',
    ax = axes[3],
    legend = False
)
axes[3].set_title('Last Exam Score by AI Usage Purpose')
axes[3].set_ylabel('Last Exam Score')
axes[3].set_xlabel('AI Usage Purpose')


# plt.figure(figsize=(4, 4))
    #plt.bar(grouped_df['ai_usage_purpose'], grouped_df['improvement_rate'], color='skyblue')
    #plt.xlabel('AI Usage Purpose')
    #plt.ylabel('Average Improvement Rate')
    #plt.title('Average Improvement Rate by AI Usage Purpose')
    #plt.xticks(rotation=45, ha = 'right')
    #plt.tight_layout()
    #plt.show()

#plt.figure(figsize=(4,4))
    #plt.bar(grouped_df['ai_usage_purpose'], grouped_df['final_score'], color = 'salmon')
    #plt.xlabel('AI Usage Purpose')
    #plt.ylabel('Average Final Score')
    #plt.title('Average Final Score by AI Usage Purpose')
    #plt.xticks(rotation=45, ha = 'right')
    #plt.tight_layout()
    #plt.show()

plt.tight_layout()
plt.show()

"""Students who use AI as an interactive academic assistant (homework, coding, doubt solving) show 
higher improvement rates and slightly stronger academic performance compared to students who use AI 
passively for note generation.


>>> AI effectiveness depends more on HOW it is used than simply WHETHER it is used.
"""

metrics = ['improvement_rate', 'last_exam_score', 'assignment_scores_avg', 'final_score']

grouped_df.set_index("ai_usage_purpose")[metrics].plot(kind = 'bar', figsize = (12,6))
plt.ylabel("Average Scores / Improvement")
plt.title("Scores & Improvement by AI Usage Purpose")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Drop NaNs
df_valid = df.dropna(subset=['ai_usage_purpose', 'improvement_rate'])

# ANOVA test
anova_result = stats.f_oneway(
    *[group['improvement_rate'].values for name, group in df_valid.groupby('ai_usage_purpose')]
    
)

print("ANOVA F-statistic:", anova_result.statistic)
print("ANOVA p-value:", anova_result.pvalue)

"""
ANOVA F-statistic: 0.7975719815478913
ANOVA p-value: 0.5265472363366541

Students within the same AI usage category differ more from each other
than the average difference between categories. That’s already a strong signal that groups are not meaningfully different.
We fail to reject the null hypothesis. The null hypothesis in ANOVA says: All group means are equal.
So the result means: There is no statistically significant difference in improvement rate across AI usage purposes.

Secondly:
The dataset suggests:
The way AI is used (purpose) does not significantly impact improvement rate, at least in this sample.
AI usage purpose may not matter as much as we think.
Or sample size may be too small.
Or effect size is small.

Note: 
Small Sample Size
If groups are small, ANOVA has low power.
Effect Is Very Small
There may be differences, but tiny ones.
High Within-Group Variation
Students vary a lot individually.
"""

print("Value Counts: ",df_valid['ai_usage_purpose'].value_counts())
print("Standard Deviations: ",df_valid.groupby('ai_usage_purpose')['improvement_rate'].std())


"""
Final Conclusion:
There is no evidence that improvement rate differs based on AI usage purpose.
This does NOT mean: AI has no effect. Or groups are exactly equal.
It means: Based on my data, we do not have enough evidence to prove a difference.
"""


plt.figure(figsize = (8,8))

df_valid.boxplot(
    column = 'improvement_rate', 
    by = 'ai_usage_purpose',
    grid = False,
    
)

plt.title("Improvement Rate by AI Usage Purpose")
plt.suptitle("")
plt.xlabel("AI Usage Purpose")
plt.ylabel("Improvement Rate")
plt.tight_layout()
plt.xticks(rotation = 45, ha = 'right')
plt.show()

"""
>>> The internal variability (SD ≈ 17) is substantially larger than the mean differences between 
groups (≈ 0.8), resulting in a very small effect size and heavy distributional overlap.

>>> The internal variability (SD ≈ 17) is substantially larger than the mean differences 
between groups (≈ 0.8), resulting in a very small effect size and heavy distributional overlap.

>>> Given:
Large sample size (~1300 per group)
High internal standard deviation (~17)
Very small mean differences (~0.8)
Non-significant ANOVA (p = 0.53)

>>> We conclude:
AI usage purpose does not meaningfully impact improvement rate in this dataset. 
Most variation in improvement is due to individual differences rather than usage category.

>>> This dataset suggests:
Improvement rate is driven more by:
- Individual ability
- Motivation
- Study behavior
- Other unmeasured factors

Not by AI usage purpose category."""

#4. Which AI tools are associated with higher improvement?
# Grouped by ai_tools_used:

grouped_ai_tools = df_valid.groupby("ai_tools_used").agg({
    'improvement_rate': 'mean',
    'last_exam_score': 'mean',
    'assignment_scores_avg': 'mean',
    'final_score': 'mean'
}).reset_index()

# print(grouped_ai_tools.head(20))
# print(grouped_ai_tools.columns)
metrics = ['improvement_rate', 'last_exam_score', 'assignment_scores_avg', 'final_score']

grouped_ai_tools.set_index("ai_tools_used")[metrics].plot(kind = 'bar', figsize = (12,6), legend = True)
plt.ylabel("AI Tools Used")
plt.title("Scores & Improvement by AI Tools Used")
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()
plt.show()

"""
On the graph it shows a uniform or constant imrpovement across all the AI tools used for all the categories. 
The type of AI used does not affect the improvement scores. 
This suggests that the specific AI tool used may not be a critical factor in driving improvement, and 
that other factors such as how the AI is used (no it doesnt because i have analysed it above ) or individual differences may play a more 
significant role in determining improvement outcomes.
"""

#Testing the f anova test to see if it really is different
df_valid_ai_tools = df.dropna(subset= ['ai_tools_used', 'improvement_rate'])
groups = [group['improvement_rate'].values for _, group in df_valid_ai_tools.groupby('ai_tools_used')]
print("Groups: ", groups)

ai_tools_anova_result =  stats.f_oneway(*groups)

print("F-Statistic: ",ai_tools_anova_result.statistic)
print("P-value: ",ai_tools_anova_result.pvalue)

#std and value counts for ai tools used
print("Value counts: ", df_valid_ai_tools['ai_tools_used'].value_counts())
print("Standard Deviations: ", df_valid_ai_tools.groupby("ai_tools_used")['improvement_rate'].std())

"""
The reason for the slight differences in the improvement rates across AI tools could be random variation or noise in the data,
especially given the high standard deviations and non-significant ANOVA result (p = 0.48)
"""
