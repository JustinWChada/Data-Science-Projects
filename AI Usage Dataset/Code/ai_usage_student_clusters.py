import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stats

dataset = pd.read_csv("ai_impact_student_performance_dataset.csv")
# dataset.loc[dataset['uses_ai'] == 0, 'ai_usage_purpose'] = np.nan
# target outcome is to predict student performance based on AI usage
# improvement : academic score final score
# metrics: time_spent, frequency of tool usage

# print(dataset.info())

ai_use_yes = dataset[dataset['uses_ai'] == 1][['final_score','performance_category']]
ai_use_no = dataset[dataset['uses_ai'] == 0][['final_score','performance_category']]


#chart to see if the people who use ai tools have better performance or not
ai_use_yes_scores_high = ai_use_yes[ai_use_yes['performance_category'] == "High" ].value_counts()
ai_use_yes_scores_middle = ai_use_yes[ai_use_yes['performance_category'] == "Middle" ].value_counts()
ai_use_yes_scores_low = ai_use_yes[ai_use_yes['performance_category'] == "Low" ].value_counts()
 
plt.figure(figsize=(10,6))
plt.hist(ai_use_yes['performance_category'], bins = 3, alpha = 0.7, label = 'AI Users', color = 'blue')
plt.hist(ai_use_no['performance_category'], bins = 3, alpha = 0.7, label = 'Non-AI Users', color = 'orange')
plt.xlabel('Performance Category')
plt.ylabel('Number of Students')
plt.title('Student Performance by AI Tool Usage')
plt.legend()
plt.show()

""" conclusion: AI Usage seems to have no impact on student performance based on the histogram above. """

ai_usage_purpose = dataset[['uses_ai', 'ai_usage_purpose', 'performance_category' ]] #.value_counts().unstack(),'final_score'
#ai_usage_purpose_yes = ai_usage_purpose.loc[1]


ai_usage_yes = ai_usage_purpose[ai_usage_purpose['uses_ai'] == 1]
ai_usage_no = ai_usage_purpose[ai_usage_purpose['uses_ai'] == 0]
print(ai_usage_no)
# Step 2: Count how many students use AI for each purpose
yes_purpose_counts = ai_usage_yes['ai_usage_purpose'].value_counts()
no_purpose_counts = ai_usage_no['ai_usage_purpose'].value_counts()
# print(purpose_counts)

# purpose_performance = ai_usage_yes.groupby(['ai_usage_purpose', 'performance_category']).size().unstack(fill_value=0)
yes_purpose_performance = pd.crosstab(ai_usage_yes['ai_usage_purpose'], ai_usage_yes['performance_category'])
yes_purpose_performance_percentage = yes_purpose_performance.div(yes_purpose_performance.sum(axis=1), axis=0) * 100 #coding tend to have higher performance categories compared to those who use AI for other purposes. 

no_purpose_performance = pd.crosstab(ai_usage_no['ai_usage_purpose'], ai_usage_no['performance_category'])
no_purpose_performance_percentage = no_purpose_performance.div(no_purpose_performance.sum(axis=1), axis=0) * 100

# coding_users = ai_usage_yes[ai_usage_yes['ai_usage_purpose'] == 'Coding']
    # doubt_users = ai_usage_yes[ai_usage_yes['ai_usage_purpose'] == 'Doubt Solving']
    # exam_users = ai_usage_yes[ai_usage_yes['ai_usage_purpose'] == 'Exam Prep']
    # homework_users = ai_usage_yes[ai_usage_yes['ai_usage_purpose'] == 'Homework']
    # notes_users = ai_usage_yes[ai_usage_yes['ai_usage_purpose'] == 'Notes']

# print(purpose_performance_percentage)

# ai_usage_purpose_yes_clean = ai_usage_purpose_yes.dropna(axis = 1, how = 'all')
# print(ai_usage_purpose_yes_clean)
#plt.figure(figsize=(12,6))
    #plt.hist(purpose_performance['performance_category'], bins = 3, alpha = 0.7, label = 'Coding Users', color=['green'])#
    # plt.hist(coding_users['performance_category'], bins = 5, alpha = 0.7, label = 'Coding Users', color=['green'])#
    # plt.hist(doubt_users['performance_category'], bins = 5, alpha = 0.7, label = 'Doubt Solving Users', color=['orange'])#, 'purple', 'brown'
    # plt.hist(exam_users['performance_category'], bins = 5, alpha = 0.7, label = 'Exam Prep Users', color = ['cyan'])
    # plt.hist(homework_users['performance_category'], bins = 5, alpha =0.7, label = 'Homework Users', color = ['magenta'])
    # plt.hist(notes_users['performance_category'], bins = 5, alpha = 0.7, label = 'Notes Users', color = ['yellow'])
    # plt.xlabel('AI Usage Purpose and Performance Category')
    # plt.ylabel('Number of Students')
    # plt.title('AI Usage Purpose and Performance Category')
    # plt.legend()
    # plt.show()
# conclusion: From the histogram, it appears that students who use AI tools for 

fig, axes = plt.subplots(1, 2, figsize = (18,6), sharey=True)

yes_purpose_performance_percentage.plot(kind='bar', stacked = True, color =['green', 'orange', 'blue'], alpha=0.7, ax =axes[0])
axes[0].set_title('AI Users')
axes[0].set_ylabel('Percentage of Students')
axes[0].set_xlabel('AI Usage Purpose')

no_purpose_performance_percentage.plot(kind='bar', stacked = True, color = ['green', 'orange', 'blue'], alpha=0.7, ax = axes[1])
axes[1].set_title('Non-AI Users')
axes[1].set_ylabel('Percentage of Students')
axes[1].set_xlabel('AI Usage Purpose')

plt.suptitle('Performance Disctribution by AI Usage Purpose (100% Stacked) Comparison between AI-Users and Non-AI-Users', y=1.02)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Performance Category", bbox_to_anchor=(1.05,0.9), loc='upper left')

plt.tight_layout(pad=3.0)
plt.show()

plt.bar(yes_purpose_performance_percentage.index, yes_purpose_performance_percentage['High'], label='High', alpha=1, color='green')
plt.bar(yes_purpose_performance_percentage.index, yes_purpose_performance_percentage['Medium'], label='Medium', alpha=1, color='orange')
plt.bar(yes_purpose_performance_percentage.index, yes_purpose_performance_percentage['Low'], label='Low', alpha=1, color='blue')
plt.xlabel('AI Usage Purpose')
plt.ylabel('Percentage of Students')
plt.title('Performance Distribution by AI Usage Purpose (100% Stacked)')
plt.legend(title='Performance Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(pad=3.0)
plt.show()

"""
- AI usage correlates with a slightly higher proportion of High performers, but the effect is modest.
- Most students, regardless of AI usage, cluster in Medium performance, showing that AI tools may help but are not a silver bullet.
- AI does not reduce the number of Low performers significantly, meaning interventions beyond AI (study strategies, teaching quality, motivation) are still crucial.
- To truly understand impact, we’d need to look at proportions within each usage purpose (e.g., does doubt solving yield more High performers than exam prep?) rather than just overall counts.
"""

#to see the proportions of performance categories within each AI usage purpose
# Plot 100% stacked bar chart
ax = yes_purpose_performance_percentage.plot(kind="bar", stacked=True, figsize=(10,6),
                     color=["#4CAF50", "#FF9800", "#2196F3"])

plt.title("Performance Distribution by AI Usage Purpose (100% Stacked)")
plt.ylabel("Percentage of Students")
plt.xlabel("AI Usage Purpose")
plt.legend(title="Performance Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

""" There are more people who passed who used AI for exam prep more than those who only used it for doubt-solving"""


for purpose in yes_purpose_performance_percentage.index:
    print(f"{purpose} users performance distribution:")
    print(yes_purpose_performance_percentage.loc[purpose])

ax = yes_purpose_performance_percentage.plot(kind="bar", stacked=True, figsize=(10,6),
                     color=["#4CAF50", "#FF9800", "#2196F3"])

for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center')

plt.title("Performance Distribution by AI Usage Purpose (100% Stacked)")
plt.ylabel("Percentage of Students")
plt.xlabel("AI Usage Purpose")
plt.legend(title="Performance Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

"""
- Uniform distribution: The performance distribution is strikingly similar across all AI usage purposes. No single purpose (e.g., Exam Prep vs. Notes) shows a strong advantage in producing higher performance outcomes.
- Neutral effect of AI purpose: Whether students use AI for coding, doubt solving, exam prep, homework, or notes, the performance outcomes remain largely unchanged.
- Possible explanation: This suggests that the way AI is used may not be the determining factor in academic success. Instead, other variables (study hours, prior preparation, motivation, or teaching quality) likely play a stronger role.

"""