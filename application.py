from flask import Flask, render_template , request
import pandas as pd
import pickle

app = Flask(__name__)
with open("LinearRegressionModel.pkl", "rb") as file:
    model = pickle.load(file)
car = pd.read_csv('cleaned_car_data.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_model = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    return render_template('index.html', companies=companies, car_model=car_model, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('model')
    year = int(request.form.get('year'))
    fuel = request.form.get('fuel')
    km_driven = int(request.form.get('km_driven'))

    predict = model.predict(pd.DataFrame([[car_model,company,year,km_driven,fuel]],columns=['name','company','year','kms_driven','fuel_type']))
    predict = int(predict)
    predict = str(predict)
    return predict

if __name__ == "__main__":
    app.run(debug=True)
