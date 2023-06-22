import plotly.graph_objects as go

models = ['AdaBoost', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost']
accuracy = [73, 74, 73, 73, 73]

fig = go.Figure(
    data=[
        go.Bar(x=models, y=accuracy, text=accuracy, textposition='auto')
    ]
)

fig.update_layout(
    title_text='Model Accuracy Comparison', 
    xaxis_title='Model',
    yaxis_title='Accuracy (%)'
)

fig.show()
