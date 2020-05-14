#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from PIL import Image


#formatting the image for processing (normalize pixels and changing shape to (224,224)
def process_image(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image,(224,224)) 
  image /= 255
  return image.numpy()


def predict(image_path, model, top_k):
  
  im = Image.open(image_path)
  image = np.asarray(im)
  image = process_image(image)
  processed_image = np.expand_dims(image,axis=0)
  
  predictions = model(processed_image, training=False)
  prob_predictions = predictions[0]


  top_k_probs, top_k_indices = tf.math.top_k(prob_predictions, k=top_k)
  probs = top_k_probs.numpy().tolist()
  classes = top_k_indices.numpy().tolist()
  classes = [n+1 for n in classes]
  return probs, classes
