import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats as stats

dataset = pd.read_csv("ai_impact_student_performance_dataset.csv")

ai_dependency = dataset[['ai_dependency_score', 'final_score']]

pearson_corr, pearson_p_value = stats.pearsonr(ai_dependency['ai_dependency_score'], ai_dependency['final_score'])

print(pearson_corr, pearson_p_value) #Indeed there is a relationship between the two
x = ai_dependency['ai_dependency_score']
y = ai_dependency['final_score']

m,b = np.polyfit(x,y,1)

plt.figure(figsize=(10,6))
plt.scatter(x, y, color ='blue', alpha = 0.7, label = '(Students)')
plt.plot(x, m*x + b, color = 'red', linewidth=2, label = "Regression Line")
plt.title("Final Exam Scores vs AI Dependency Score", fontsize = 16)
plt.xlabel("AI Dependency Score", fontsize = 16)
plt.ylabel("Final Exam Scores", fontsize= 16)
plt.grid(True)
plt.legend(title="AI Dependenccy Score vs Final Score")
plt.show()

#The graph and statistics together indicate essentially no meaningful linear relationship between AI dependency score and final exam scores.

