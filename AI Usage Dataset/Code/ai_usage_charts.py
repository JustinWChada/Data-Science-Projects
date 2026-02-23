import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as stats

dataframe = pd.read_csv('ai_impact_student_performance_dataset.csv')

def analyze_exam_scores_vs_study_time(dataframe):
    plt.figure(figsize=(10,10))
    plt.scatter(dataframe['study_hours_per_day'], dataframe['final_score'], color = 'blue', alpha=0.5)
    plt.title('Final Exam Scores vs Study Hours Per Day', fontsize=16)
    plt.xlabel('Study Hours Per Day', fontsize = 14)
    plt.ylabel('Final Exam Scores', fontsize = 14)
    plt.grid(True)
    plt.show()

def analyze_exam_scores_vs_ai_usage_time(dataframe):
    plt.figure(figsize=(10,10))
    plt.scatter(dataframe['ai_usage_time_minutes'], dataframe['final_score'], color = 'green', alpha=0.5)
    plt.title('Final Exam Scores vs AI Usage Time (in Minutes)', fontsize=16)
    plt.xlabel('AI Usage Time (in Minutes)', fontsize = 14)
    plt.ylabel('Final Exam Scores', fontsize = 14)
    plt.grid(True)
    plt.show()

# uses_ai = pd.DataFrame({
#     'count': dataframe['uses_ai'].value_counts(),
#     'percentage': dataframe['uses_ai'].value_counts(normalize = True) * 100
# })

exam_scores = pd.DataFrame({
    'uses_ai': dataframe['uses_ai'],
    'final_score': dataframe['final_score']
})

# uses_ai_yes_counts = exam_scores[exam_scores['uses_ai'] == 1]['final_score'].value_counts()
# uses_ai_no_counts = exam_scores[exam_scores['uses_ai'] == 0]['final_score'].value_counts()

# print(uses_ai_yes_counts)
# print(uses_ai_no_counts)

def plot_uses_ai_vs_final_score(dataframe):
    uses_ai_yes = dataframe[dataframe['uses_ai'] == 1]['final_score']
    uses_ai_no = dataframe[dataframe['uses_ai'] == 0]['final_score']

    plt.figure(figsize=(10,10))
    plt.hist([uses_ai_yes, uses_ai_no], bins=15, alpha=0.7, color=['blue', 'red'], label=['Uses AI', 'Does not use AI'])
    plt.title('Final Exam Scores vs AI Use (Yes/No)', fontsize=16)
    plt.xlabel('Final Exam Scores', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_exam_scores_vs_uses_ai(dataframe):
    plt.figure(figsize = (10,10))
    plt.scatter(dataframe['uses_ai'],dataframe['final_score'], color = 'orange', alpha = 0.7)
    plt.title('Final Exam Scores vs AI Use (Yes/No)', fontsize = 16)
    plt.xlabel('AI Use (Yes/No)', fontsize = 12)
    plt.ylabel('Final Exam Scores', fontsize = 12)
    plt.grid(True)
    plt.show()

plot_uses_ai_vs_final_score(dataframe)
analyze_exam_scores_vs_study_time(dataframe)
analyze_exam_scores_vs_ai_usage_time(dataframe)
analyze_exam_scores_vs_uses_ai(dataframe)
# uses_ai_yes = dataframe[dataframe['uses_ai'] == 1]['final_score']
# uses_ai_no = dataframe[dataframe['uses_ai'] == 0]['final_score']

# print(uses_ai_yes, uses_ai_no)

"""
Conclusion: 
>>> From the bar and scatter plot, it appears that there is no significant difference in final exam scores between students who use AI tools and those who do not. The distribution of final scores for both groups seems to be similar, suggesting that AI usage may not have a strong impact on student performance in this dataset. However, further statistical analysis would be needed to confirm this observation.
>>> It is also observed that there seems to be no clear correlation between study hours per day and final exam scores, 
>>> as well as between AI usage time and final exam scores. The scatter plots do not show a strong positive or negative trend, indicating that other factors may be influencing student performance more significantly than study hours or AI usage time.
""" 