from flask import Flask, request, jsonify
import joblib

# Load the saved pipeline/model
model = joblib.load(r"C:\Users\Raneem\Desktop\sentiment_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Sentiment Analysis API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json(force=True)
        tweet = data.get("tweet", "")

        if not tweet:
            return jsonify({"error": "No tweet provided"}), 400

        # Make prediction
        prediction = model.predict([tweet])[0]

        # Return response
        return jsonify({
            "tweet": tweet,
            "sentiment": str(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)