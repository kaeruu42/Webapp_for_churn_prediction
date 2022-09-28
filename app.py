
import numpy as np
from flask import Flask, request, render_template, flash
import pickle

# Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder='template')
app.config['SECRET_KEY'] = 'skjdfhwiufbskdfj'


# create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 26)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

# when form is submitted
@app.route('/', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        #return render_template('output.html', prediction_text=result)
        if result == 0:
            return render_template('output.html', prediction_text="Predicted to not churn.")
        elif result == 1:
            return render_template('output.html', prediction_text='Predicted to churn.')


if __name__ == "__main__":
    app.run(debug=True)