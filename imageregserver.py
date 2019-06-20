import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Model
import PIL
import base64
import io
import socket

from flask import request
from flask import Flask
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'

def siapkan_model():
    global model
    global graph
    model = load_model('predictproduct5.h5')
    print("Model Sudah Siap!")
    graph = tf.get_default_graph()

def prepare_image(file):
    img_array = image.img_to_array(file)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

siapkan_model()

@app.route('/kenali', methods=['POST'])
@cross_origin()
def kenali():
    with graph.as_default():
        object_masuk = request.get_json(force=True)
        encoded = object_masuk['image']
        decoded = base64.b64decode(encoded)
        img = image.load_img(io.BytesIO(decoded),target_size=(224,224))
        processed_image = prepare_image(img)
        prediksi = model.predict(processed_image)
        response = {'Prediksi':
            {'Ash Brown': prediksi[0][0]*100,
             'Ash Purple': prediksi[0][1]*100,
             'Bye Bye Fever': prediksi[0][2]*100,
             'Dinh Lang': prediksi[0][3]*100,
             'Oillan Baby': prediksi[0][4]*100,
             'Oillan Mama': prediksi[0][5]*100,
             'Omepros 10': prediksi[0][6]*100,
             'Omepros 30': prediksi[0][7]*100,
             'Tho Phuc Linh': prediksi[0][8]*100,
	     'Hostname' : socket.gethostname()}}
    return json.dumps(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
