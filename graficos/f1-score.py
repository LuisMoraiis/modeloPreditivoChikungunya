import matplotlib.pyplot as plt
import numpy as np

modelos = ['Adaboost', 'DecisionTree', 'RandomForest', 'Logistic Regression', 'xgboost']
f1_scores_class0 = np.array([0.72, 0.72, 0.72, 0.72, 0.72])  
f1_scores_class1 = np.array([0.74, 0.75, 0.74, 0.74, 0.75])  

average_f1_scores = (f1_scores_class0 + f1_scores_class1) / 2

fig, ax = plt.subplots()

bar_positions = np.arange(len(modelos))
bar_widths = average_f1_scores
bars = ax.barh(bar_positions, bar_widths, align='center')

ax.set_yticks(bar_positions)
ax.set_yticklabels(modelos)

ax.set_xlabel('Average F1-Score')
ax.set_ylabel('Modelos')
ax.set_title('Comparação de Desempenho dos Modelos')

ax.set_xlim([0.7, 0.76])

for bar, score in zip(bars, average_f1_scores):
    ax.text(score, bar.get_y() + bar.get_height() / 2,
            f'{score:.2f}', 
            va='center', ha='left', color='black')

plt.show()

