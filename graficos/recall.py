import matplotlib.pyplot as plt
import numpy as np

modelos = ['Adaboost', 'DecisionTree', 'RandomForest', 'Logistic Regression', 'xgboost']
recall_class0 = np.array([0.68, 0.67, 0.68, 0.68, 0.67])
recall_class1 = np.array([0.79, 0.80, 0.79, 0.79, 0.80])

average_recall = (recall_class0 + recall_class1) / 2

fig, ax = plt.subplots()

bar_positions = np.arange(len(modelos))
bar_widths = average_recall
bars = ax.barh(bar_positions, bar_widths, align='center')

ax.set_yticks(bar_positions)
ax.set_yticklabels(modelos)

ax.set_xlabel('Average Recall')
ax.set_ylabel('Modelos')
ax.set_title('Comparação de Desempenho dos Modelos')

ax.set_xlim([0.6, 0.85])

for bar, score in zip(bars, average_recall):
    ax.text(score, bar.get_y() + bar.get_height() / 2,
            f'{score:.2f}', 
            va='center', ha='left', color='black')

plt.show()
