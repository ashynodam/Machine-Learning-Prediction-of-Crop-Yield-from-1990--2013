 # 1. Importing Libraries

from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

# lask: Framework for creating the web application.
# request: For handling form data submitted via the web interface.
# render_template: For rendering HTML templates.
# NumPy: For creating and manipulating numerical arrays.
# pickle: For loading the pre-trained machine learning model and preprocessing pipeline.
# sklearn: To check the version of scikit-learn used for compatibility.

# 2. Loading the Pre-Trained Models

dtr = pickle.load(open('C:/Users/Gbenga AKINDEKO/Documents/dtr.pkl', 'rb'))
preprocessor = pickle.load(open('C:/Users/Gbenga AKINDEKO/Documents/preprocesser.pkl', 'rb'))

# dtr.pkl: Contains the trained Decision Tree Regressor model for prediction.
# preprocessor.pkl: Contains the preprocessing pipeline (e.g., encoding categorical variables, scaling numeric data).

# 3. Creating the Flask Application

new_app = Flask(__name__) # Initializes a Flask app object

# 4. Defining the Routes
@new_app.route('/')
def index():
    return render_template('index.html')

# /: The home page route that renders index.html.
# index.html: The HTML template where users can input the required data.

# 5. Prediction Route

@new_app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

# /predict: The route for handling predictions. It accepts POST requests when users submit data through the form.
# request.form: Retrieves user inputs from the submitted form.

# 6. Processing User Input
        
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)

# features: Creates a NumPy array from the user's inputs (in object dtype to handle mixed types).
# preprocessor.transform: Transforms the input data using the pre-trained preprocessing pipeline (e.g., scales numerical values, encodes categorical variables).

# 7. Generating Prediction
        
        prediction = dtr.predict(transformed_features).reshape(1, -1)

# dtr.predict: Predicts the yield based on the processed input.
# reshape(1, -1): Ensures the output is in the correct shape for returning to the user.

# 8. Returning the Prediction

        return render_template('index.html', prediction=prediction[0][0])


if __name__ == "__main__":
    new_app.run(debug=True)



