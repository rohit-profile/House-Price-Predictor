from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd  

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/house_model.h5")

# Load feature scaler
scaler_x = pickle.load(open("model/scaler_x.pkl", "rb"))

# Load target scaler
scaler_y = pickle.load(open("model/scaler_y.pkl", "rb"))  # ✅ Added


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric inputs
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        parking = int(request.form['parking'])

        mainroad  = 1 if request.form['mainroad'] == 'yes' else 0
        guestroom = 1 if request.form['guestroom'] == 'yes' else 0
        basement = 1 if request.form['basement'] == 'yes' else 0
        hotwaterheating = 1 if request.form['hotwaterheating'] == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0

        # Use DataFrame for correct column names
        input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories,
                                    mainroad, guestroom, basement,
                                    hotwaterheating, airconditioning, parking]],
                                  columns=['area', 'bedrooms', 'bathrooms', 'stories',
                                           'mainroad', 'guestroom', 'basement',
                                           'hotwaterheating', 'airconditioning', 'parking'])

        # Scale input features
        input_scaled = scaler_x.transform(input_data)

        # Predict (without progress bar)
        prediction_scaled = model.predict(input_scaled, verbose=0)

        #  Convert scaled prediction to real house price

        predicted_price = scaler_y.inverse_transform(prediction_scaled)[0][0]

        return render_template('index.html',
                               prediction_text=f'Predicted House Price (₹) : {predicted_price:.2f}')

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {e}')


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
