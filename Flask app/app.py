from flask import Flask,jsonify,request
import joblib
import pandas as pd
from flask.templating import render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict/",methods=['GET'])

def predict():
    result=request.args
    model_type=''
    data=[[int(result['age']),int(result['number']),int(result['start'])]]
    if(result['model']=='dt'):
        model_type='Decision Tree'
        model=joblib.load('Decision_tree_model.sav')
        prediction=model.predict(data)
    else:
        model_type='Random Forest Classifier'
        model=joblib.load('Random_forest_classifier_model.sav')
        prediction=model.predict(data)
    return jsonify({'model':model_type,'kyphosis_status': prediction[0]}) 


if __name__ == '__main__':
    app.run()