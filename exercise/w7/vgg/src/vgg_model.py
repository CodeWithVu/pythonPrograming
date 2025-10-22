from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

def create_vgg_model():
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def load_pretrained_weights(model, weights_path):
    model.load_weights(weights_path)

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    return predictions

def decode_predictions(predictions):
    from tensorflow.keras.applications.vgg16 import decode_predictions
    return decode_predictions(predictions, top=5)[0]