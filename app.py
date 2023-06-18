from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from src.data_pipeline.predict_pipeline_scripts import CustomData,Predict_pipeline

app = Flask(__name__, template_folder='templates')

@app.route('/',methods = ['GET', 'POST'])
def predict_results():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        data = CustomData(
            request.form.get('brandSlogan'),
            request.form.get('hasVideo'),
            request.form.get('rating'),
            request.form.get('priceUSD'),
            request.form.get('countryRegion'),
            request.form.get('startDate'),
            request.form.get('endDate'),
            request.form.get('teamSize'),
            request.form.get('hasGithub'),
            request.form.get('hasReddit'),
            request.form.get('platform'),
            request.form.get('coinNum'),
            request.form.get('minInvestment'),
            request.form.get('distributedPercentage'),
        )
        prediction = Predict_pipeline()
        prediction_result = prediction.predict(data)

        return render_template('home.html', result = prediction_result)


if __name__ == '__main__':
    app.run(debug=True)