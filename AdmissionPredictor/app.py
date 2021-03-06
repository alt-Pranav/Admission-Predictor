#Source:
# https://www.analyticsvidhya.com/blog/2020/09/integrating-machine-learning-into-web-applications-with-flask/

#importing libraries
import numpy as np
from flask import Flask, render_template, request
import pickle

#initialise the Flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# default page of web-app
@app.route('/')
def home():
    return render_template('index.html')

# to use the predict button in the web app
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Chance of Admission = {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)