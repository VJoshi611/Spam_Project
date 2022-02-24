import numpy as np
from flask import Flask, request, jsonify, render_template  #render_template is for rendering html template
import pickle   # to read pickle file


app = Flask(__name__)  # flask app gets created  # initialzing flask object
model = pickle.load(open('spam.pkl', 'rb'))  # loading model

@app.route('/')    #goes to main directory , home page method 
def home():    # your task func
    return render_template('index.html')   # displays / return experience,test_Score,interview_score from index.html
#index.html contains front page html code


@app.route('/predict',methods=['POST']) # predict method. /predict will hit to def predict() func
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text_features = [x for x in request.form.values()]  # request takes all feature attributes 
    final_features = [np.array(text_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Given message is  {}'.format(output))
    # prediction_text gets replaced with {{ prediction_text }}


if __name__ == "__main__":   # if true then run the app i.e. calling app
    app.run(debug=True)