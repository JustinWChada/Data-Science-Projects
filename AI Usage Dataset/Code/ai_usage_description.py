import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as stats

dataframe = pd.read_csv('ai_impact_student_performance_dataset.csv')

#getting the mean, median, mode for exam scores
mean_score_exam = dataframe['last_exam_score'].mean()
meadian_score_exam = dataframe['last_exam_score'].median()
mode_score_exam = dataframe['last_exam_score'].mode()

print(f"Exam Scores -> Mean: {mean_score_exam}, Median: {meadian_score_exam}, Mode: {mode_score_exam[0]}")

#getting the mean, median, mode for final scores
mean_score_final_exam = dataframe['final_score'].mean()
meadian_score_final_exam = dataframe['final_score'].median()
mode_score_final_exam = dataframe['final_score'].mode()

print(f"Final Scores -> Mean: {mean_score_final_exam}, Median: {meadian_score_final_exam}, Mode: {mode_score_final_exam[0]}")

#getting the mean, median, mode for ai dependency scores
mean_score_ai_dependency = dataframe['ai_dependency_score'].mean()
meadian_score_ai_dependency = dataframe['ai_dependency_score'].median()
mode_score_ai_dependency = dataframe['ai_dependency_score'].mode()

print(f"AI Dependency Scores -> Mean: {mean_score_ai_dependency}, Median: {meadian_score_ai_dependency}, Mode: {mode_score_ai_dependency[0]}")

#getting the mean, median, mode for ai usage minutes
mean_minutes_ai_usage = dataframe['ai_usage_time_minutes'].mean()
meadian_minutes_ai_usage = dataframe['ai_usage_time_minutes'].median()
mode_minutes_ai_usage = dataframe['ai_usage_time_minutes'].mode()

print(f"AI Usage Minutes -> Mean: {mean_minutes_ai_usage}, Median: {meadian_minutes_ai_usage}, Mode: {mode_minutes_ai_usage[0]}")

#getting the mean, median, mode for study hours per day
mean_hours_daily_study_time = dataframe['study_hours_per_day'].mean()
meadian_hours_daily_study_time = dataframe['study_hours_per_day'].median()
mode_hours_daily_study_time = dataframe['study_hours_per_day'].mode()

print(f"Study Hours Per Day -> Mean: {mean_hours_daily_study_time}, Median: {meadian_hours_daily_study_time}, Mode: {mode_hours_daily_study_time[0]}")

print()

#getting the variance of final_exam_scores, final_scores, ai_dependency_scores, ai_usage_time_minutes, study_hours_per_day
variance_last_exam_scores = np.var(dataframe['last_exam_score']) # or dataframe['last_exam_score'].var()
variance_final_scores = np.var(dataframe['final_score']) # or dataframe['final_score'].var()
variance_ai_dependency_scores = np.var(dataframe['ai_dependency_score']) # or dataframe['ai_dependency_score'].var()
variance_ai_usage_time_minutes = np.var(dataframe['ai_usage_time_minutes']) # or dataframe['ai_usage_time_minutes'].var()
variance_study_hours_per_day = np.var(dataframe['study_hours_per_day']) # or dataframe['study_hours_per_day'].var()

print(f"Variance of Last Exam Scores: {variance_last_exam_scores}")
print(f"Variance of Final Scores: {variance_final_scores}")
print(f"Variance of AI Dependency Scores: {variance_ai_dependency_scores}")
print(f"Variance of AI Usage Time (in Mins): {variance_ai_usage_time_minutes}")
print(f"Variance of Study Hours Per Day: {variance_study_hours_per_day}")

print()

#getting the range for final_exam_scores, final_scores, ai_dependency_scores, ai_usage_time_minutes,_study_hours_per_day
range_last_exam_scores = np.ptp(dataframe['last_exam_score']) #dataframe['last_exam_score'].max() - dataframe['last_exam_score'].min()
range_final_exam_scores = np.ptp(dataframe['final_score'])
range_ai_dependency_scores = np.ptp(dataframe['ai_dependency_score'])
range_ai_usage_time_minutes = np.ptp(dataframe['ai_usage_time_minutes'])
range_study_hours_per_day = np.ptp(dataframe['study_hours_per_day'])

print(f"Range of Last Exam Scores: {range_last_exam_scores}")
print(f"Range of Final Exam Scores: {range_final_exam_scores}")
print(f"Range of AI Dependency Scores: {range_ai_dependency_scores}")
print(f"Range of AI Usage Time (in Mins): {range_ai_usage_time_minutes}")
print(f"Range of Study Hours Per Day: {range_study_hours_per_day}")

print()

#getting the min and max for final_exam_scores, final_scores, ai_dependency_scores, ai_usage_time_minutes, study_hours_per_day
min_last_exam_scores = dataframe['last_exam_score'].min()
max_last_exam_scores = dataframe['last_exam_score'].max()

min_final_scores = dataframe['final_score'].min()
max_final_scores = dataframe['final_score'].max()

min_ai_dependency_scores = dataframe['ai_dependency_score'].min()
max_ai_dependency_scores = dataframe['ai_dependency_score'].max()

min_ai_usage_time_minutes = dataframe['ai_usage_time_minutes'].min()
max_ai_usage_time_minutes = dataframe['ai_usage_time_minutes'].max()

min_study_hours_per_day = dataframe['study_hours_per_day'].min()
max_study_hours_per_day = dataframe['study_hours_per_day'].max()

print(f"Min of Last Exam Scores: {min_last_exam_scores}")
print(f"Max of Last Exam Scores: {max_last_exam_scores}")
print(f"Min of Final Exam Scores: {min_final_scores}")
print(f"Max of Final Exam Scores: {max_final_scores}")
print(f"Min of AI Dependency Scores: {min_ai_dependency_scores}")
print(f"Max of AI Dependency Scores: {max_ai_dependency_scores}")
print(f"Min of AI Usage Time (in Mins): {min_ai_usage_time_minutes}")
print(f"Max of AI Usage Time (in Mins): {max_ai_usage_time_minutes}")
print(f"Min of Study Hours Per Day: {min_study_hours_per_day}")
print(f"Max of Study Hours Per Day: {max_study_hours_per_day}")

print()

#getting the interquartile range for final_exam_scores, final_scores, ai_dependency_scores, ai_usage_time_minutes, study_hours_per_day
iqr_last_exam_scores = stats.iqr(dataframe['last_exam_score'])
iqr_final_scores = stats.iqr(dataframe['final_score'])
iqr_ai_dependency_scores = stats.iqr(dataframe['ai_dependency_score'])
iqr_ai_usage_time_minutes = stats.iqr(dataframe['ai_usage_time_minutes'])
iqr_study_hours_per_day = stats.iqr(dataframe['study_hours_per_day'])

print(f"IQR of Last Exam Scores: {iqr_last_exam_scores}")
print(f"IQR of Final Exam Scores: {iqr_final_scores}")
print(f"IQR of AI Dependency Scores: {iqr_ai_dependency_scores}")
print(f"IQR of AI Usage Time (in Mins): {iqr_ai_usage_time_minutes}")
print(f"IQR of Study Hours Per Day: {iqr_study_hours_per_day}")

#plotting the boxplot of all these exam_Score, final_Score, dependency_Score, usage_time_minutes, study_hours_per_Day

plt.figure(figsize=(10,6))
plt.boxplot([dataframe['last_exam_score'], dataframe['final_score'], dataframe['ai_dependency_score'], dataframe['ai_usage_time_minutes'], dataframe['study_hours_per_day']], labels=['Exam Scores', 'Final Scores', 'AI Dependency Scores', 'AI Usage Time (in Mins)', 'Study Hours Per Day'])
plt.title('Boxplot of All Scores and Study Hours')
plt.show()

print()

#getting frequency tables for categorical columns: gender and ai usage purpose categories

frequency_gender = dataframe['gender'].value_counts()
frequency_ai_usage_purpose = dataframe['ai_usage_purpose'].value_counts()

percent_frequency_gender = dataframe['gender'].value_counts(normalize= True) * 100
percent_frequency_ai_usage_purpose = dataframe['ai_usage_purpose'].value_counts(normalize = True) *100

table_frequency_gender = pd.DataFrame({
    'count': dataframe['gender'].value_counts(),
    'percentage': dataframe['gender'].value_counts(normalize = True) * 100
})

table_frequency_ai_usage_purpose = pd.DataFrame({
    'count': dataframe['ai_usage_purpose'].value_counts(),
    'percentage': dataframe['ai_usage_purpose'].value_counts(normalize = True) * 100
})

print("\nGender Frequeny Table: \n", table_frequency_gender)
print("\nAI Usage Purpose Frequency Table: \n", table_frequency_ai_usage_purpose)

print()

#getting the skewness for final_exam_scores, final_scores, ai_dependency_scores, ai_usage_time_minutes, study_hours_per_day
skweness_exam_score = stats.skew(dataframe['last_exam_score'])
skweness_final_score = stats.skew(dataframe['final_score'])
skweness_ai_dependency_score = stats.skew(dataframe['ai_dependency_score'])
skweness_ai_usage_time_minutes = stats.skew(dataframe['ai_usage_time_minutes'])
skweness_study_hours_per_day = stats.skew(dataframe['study_hours_per_day'])

print(f"Skewness of Last Exam Scores: {skweness_exam_score}")
print(f"Skewness of Final Scores: {skweness_final_score}")
print(f"Skewness of AI Dependency Scores: {skweness_ai_dependency_score}")
print(f"Skewness of AI Usage Time (in Mins): {skweness_ai_usage_time_minutes}")
print(f"Skewness of Study Hours Per Day: {skweness_study_hours_per_day}")