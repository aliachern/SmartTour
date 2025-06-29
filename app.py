from flask import Flask, request, jsonify

app = Flask(__name__)

# Dummy recommendation function for example
def get_recommendations(trip_type, budget):
    # You can replace this with real ML logic
    if trip_type == "adventure":
        if budget == "low":
            return ["Mount Ledang", "Taman Negara"]
        elif budget == "medium":
            return ["Kinabalu Park", "Endau Rompin"]
        else:
            return ["Gunung Mulu", "Sipadan Island"]
    elif trip_type == "cultural":
        return ["George Town", "Malacca", "Islamic Arts Museum"]
    elif trip_type == "relaxation":
        return ["Langkawi", "Redang Island", "Pangkor Laut"]
    else:
        return ["Kuala Lumpur", "Cameron Highlands"]

@app.route("/")
def home():
    return "SmartTour API is running."

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    trip_type = data.get("tripType")
    budget = data.get("budget")
    recommendations = get_recommendations(trip_type, budget)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)
