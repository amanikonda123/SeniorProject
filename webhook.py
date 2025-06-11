from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/my-webhook', methods=['POST'])
def my_webhook():
    # Bright Data will POST the scraped data here as JSON
    data = request.get_json()
    if data is None:
        print("No JSON received")
        return jsonify({"error": "No JSON body"}), 400

    print("Received data from Bright Data:", data)

    # Do whatever you need with the data (store it, process it, etc.)
    # For example, write to a file:
    with open("bright_data_reviews.json", "w") as f:
        json.dump(data, f)

    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(port=5050, debug=True)
