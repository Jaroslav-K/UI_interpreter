# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from flask import Flask, Blueprint, jsonify, request, send_file
from objects import feature
from nn import network
from service import service
import os
from flask_cors import CORS, cross_origin
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/api", methods=['POST'])
def update_features():
    data = request.get_json()
    prediction = service.update_features(data)

    return jsonify(prediction)


@app.route("/api", methods=['GET'])
def get_features():
    dictionary = [img_feature.to_dict() for img_feature in service.get_features()]
    return jsonify(dictionary)


@app.route("/prediction", methods=['GET'])
def get_prediction():
    prediction = service.get_prediction()
    return jsonify(prediction)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'data.jpg'))
    result = service.upload_image('data.jpg')

    return jsonify(success=result)


@app.route('/get_image', methods=['GET'])
def get_image():
    return send_file('data.jpg', mimetype='image/jpeg')


@app.route('/get_image_reconstructed', methods=['GET'])
def get_image_reconstructed():
    return send_file('reconstructed.png', mimetype='image/jpeg')

