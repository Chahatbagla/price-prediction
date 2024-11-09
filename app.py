from flask import Flask, render_template, request
import pandas as pd
import joblib

# Function to clean price strings (remove '₹' symbol and commas, and convert to float)
def clean_price(price):
    # Check if the price is a string and contains the '₹' symbol
    if isinstance(price, str):
        return float(price.replace('₹', '').replace(',', '').strip())
    # If the price is already a float, just return it
    return float(price)

# Function to ensure numeric values for ratings and no_of_ratings
def clean_ratings(rating):
    try:
        return float(rating)
    except ValueError:
        return 0.0  # return 0 if the value can't be converted to float

# Load pre-trained models
rf_model = joblib.load("random_forest_model_pipeline.pkl")
xgb_model = joblib.load("xgboost_model_pipeline.pkl")
gb_model = joblib.load("gradient_boosting_model.pkl")

# Load dataset
data = pd.read_csv("your_data.csv")

# Clean the 'actual_price', 'ratings', and 'no_of_ratings' columns in the dataset
data['actual_price'] = data['actual_price'].apply(clean_price)
data['ratings'] = data['ratings'].apply(clean_ratings)
data['no_of_ratings'] = data['no_of_ratings'].apply(clean_ratings)

# Initialize Flask app
app = Flask(__name__)

# Route to render the home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        product_name = request.form["product_name"]

        # Find the product details based on the name
        product = data[data['name'].str.contains(product_name, case=False, na=False)].iloc[0]
        product_details = {
            "name": product["name"],
            "image": product["image"],
            "ratings": product["ratings"],
            "no_of_ratings": product["no_of_ratings"],
            "actual_price": product["actual_price"]
        }

        # Prepare input for prediction
        features = pd.DataFrame([{
            "ratings": product["ratings"],
            "no_of_ratings": product["no_of_ratings"],
            "actual_price": product["actual_price"]
        }])

        # Predict using all three models
        rf_pred = rf_model.predict(features)[0]
        xgb_pred = xgb_model.predict(features)[0]
        gb_pred = gb_model.predict(features)[0]  # For Gradient Boosting model

        return render_template("index.html", product_details=product_details, rf_pred=rf_pred, xgb_pred=xgb_pred, gb_pred=gb_pred)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
