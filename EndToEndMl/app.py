import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

# create flask app
app = Flask(__name__) # starting point of application

#load pickle file
regression_model = pickle.load(open('regressionModel.pkl' , 'rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

# this is to check on post man
@app.route('/predict_api' , methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    
    # our model is trained on 2D but input is 1D. so reshape
    print(np.array(list(data.values())).reshape(1,-1))
    
    # standardize the data 
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    
    # perform prediction
    output = regression_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

# this is for front end
@app.route('/predict' , methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]

    # our model is trained on 2D but input is 1D. so reshape
    new_data = np.array(data).reshape(1,-1)

    # standardize the data 
    final_input = scalar.transform(new_data)
    print(final_input)

    # perform prediction
    output = regression_model.predict(final_input)[0]

    return render_template("home.html",prediction_text = "The House price prediction is {}".format(output))


# run application
if __name__ == "__main__":
    app.run(debug = True)