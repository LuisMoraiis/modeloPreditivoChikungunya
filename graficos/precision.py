import matplotlib.pyplot as plt
import numpy as np

modelos = ['Adaboost', 'DecisionTree', 'RandomForest', 'Logistic Regression', 'xgboost']
precision_class0 = np.array([0.77, 0.78, 0.77, 0.77, 0.78])
precision_class1 = np.array([0.71, 0.70, 0.70, 0.71, 0.70])

average_precision = (precision_class0 + precision_class1) / 2

fig, ax = plt.subplots()

bar_positions = np.arange(len(modelos))
bar_widths = average_precision
bars = ax.barh(bar_positions, bar_widths, align='center')

ax.set_yticks(bar_positions)
ax.set_yticklabels(modelos)

ax.set_xlabel('Average Precision')
ax.set_ylabel('Modelos')
ax.set_title('Comparação de Desempenho dos Modelos')

ax.set_xlim([0.6, 0.8])

for bar, score in zip(bars, average_precision):
    ax.text(score, bar.get_y() + bar.get_height() / 2,
            f'{score:.2f}', 
            va='center', ha='left', color='black')

plt.show()
