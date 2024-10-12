from flask import Flask, render_template, request, redirect
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
import io
import base64
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'datafile' not in request.files:
            return redirect(request.url)

        file = request.files['datafile']
        if file.filename == '':
            return redirect(request.url)

        # Read the uploaded CSV file
        if file:
            df = pd.read_csv(file)
            years = int(request.form['years'])

            # Preprocess the data: Assuming 'Date' column is present and financial data is in 'Revenue'
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Train the ARIMA model on historical revenue data
            model = ARIMA(df['Revenue'], order=(5, 1, 0))  # Example ARIMA(5,1,0)
            model_fit = model.fit()

            # Forecast for the next X years
            forecast = model_fit.forecast(steps=years)

            # Create the interactive plot using Plotly
            historical_trace = go.Scatter(x=df.index, y=df['Revenue'], mode='lines', name='Historical Revenue')
            future_trace = go.Scatter(x=pd.date_range(start=df.index[-1], periods=years+1, freq='Y')[1:], y=forecast, mode='lines', name='Forecasted Revenue')

            layout = go.Layout(
                title=f'Revenue Prediction for Next {years} Years',
                xaxis=dict(title='Year'),
                yaxis=dict(title='Revenue'),
            )
            fig = go.Figure(data=[historical_trace, future_trace], layout=layout)

            # Convert the Plotly figure to HTML
            plot_div = fig.to_html(full_html=False)

            # SHAP Explanation for ARIMA
            shap_values = model_fit.resid

            # Create the SHAP-like plot using residuals
            shap_fig, ax = plt.subplots()
            plt.plot(df.index, shap_values, label='Residuals')
            plt.title('Residual Plot')
            plt.legend()

            # Save the SHAP plot to a buffer
            buf = io.BytesIO()
            shap_fig.savefig(buf, format='png')
            buf.seek(0)
            shap_plot_url = base64.b64encode(buf.getvalue()).decode()

            # Prediction accuracy explanation
            prediction_accuracy = f"""
            The predicted results are based on the company's historical performance. 
            If the company maintains its current progress and market conditions remain stable, 
            the forecast could be highly accurate. However, unforeseen factors such as 
            economic downturns, regulatory changes, or internal company issues could significantly 
            impact the actual revenue, leading to deviations from the predicted values.
            """

            return render_template('index.html', plot_div=plot_div, shap_plot_url=shap_plot_url, explanation=prediction_accuracy)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
