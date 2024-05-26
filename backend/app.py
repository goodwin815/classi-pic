from flask import Flask, request, jsonify
from flask_cors import CORS
from engine.classify import Classify

app = Flask(__name__)
CORS(app)

# Initialize the Classify class
classifier = Classify()

@app.route('/classify', methods=['POST'])
def classify_image():
    res = classifier.classify_image()
    return jsonify({'class': res })

if __name__ == '__main__':
    app.run(debug=True)