#!/usr/bin/python3

import argparse
import json as simplejson
from utils import *
from PIL import Image
import sys

def Main():
    #parse out variables
    parser = argparse.ArgumentParser(description='Image Classifier.')

    #mandatory arguments 
    parser.add_argument('path', help='string; filepath of image')
    parser.add_argument('model', help='.h5 file')
    
    #not mandatory arguments
    parser.add_argument('-t'l,'--top_k', help='integer; the number of top responses',action='store_true')
    parser.add_argument('c','--category_names', help='a json file; map of label to catetgory',action='store_true',default='./label_map.json')

    args = parser.parse_args()
    
    predict_probs(args)
    
    if args.top_k:
        top_k_probs, top_k_indices = tf.math.top_k(prob_predictions, k=args.top_k)
        probs = top_k_probs.numpy().tolist()
        classes = top_k_indices.numpy().tolist()
        classes = [n+1 for n in classes]
        labels = [class_names[str(n+1)] for n in classes]

        for _ in range(args.top_k):
            print('\t\u2022' + str(probs[_]) + ':' + str(labels[_]))
            
    else:
        top_k_probs, top_k_indices = tf.math.top_k(prob_predictions, k=5)
        probs = top_k_probs.numpy().tolist()
        classes = top_k_indices.numpy().tolist()
        classes = [n+1 for n in classes]
        labels = [class_names[str(n+1)] for n in classes]

        print(labels[np.argmax(probs)],max(probs))

        
#creating the model
#loading the MobileNet from TensorFlow Hub
url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

feature_extractor = hub.KerasLayer(url, input_shape = (image_size, image_size,3))
feature_extractor.trainable = False

# build the model
model = tf.keras.Sequential([feature_extractor, tf.keras.layers.Dense(102, activation='softmax')])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

#loading weights
model.load_weights(args.model)
  
#map labels
with open(args.category_names,'r') as f:
    class_names = simplejson.load(f)

#process image
def process_image(image_path):
    #process image
    im = Image.open(image_path)
    image = np.asarray(im)
    image = process_image(image)
    processed_image =  np.expand_dims(image,axis=0)
    return processed_image
    
def predict_probs(args):
    #make predictions
    predictions = model(process_image(args.path), training=False)
    prob_predictions = predictions[0]
    return prob_predictions
    
    #outputs

#     #print the top_k and their associated probabilities
#     if args.top_k != 5:
        
#         top_k_probs, top_k_indices = tf.math.top_k(prob_predictions, k=args.top_k)
#         probs = top_k_probs.numpy().tolist()
#         classes = top_k_indices.numpy().tolist()
#         classes = [n+1 for n in classes]
#         labels = [class_names[str(n+1)] for n in classes]

#         for _ in range(args.top_k):
#             print('\t\u2022' + str(probs[_]) + ':' + str(labels[_]))
            
#     #print the most likely label & it's associated probability
#     else:
#         top_k_probs, top_k_indices = tf.math.top_k(prob_predictions, k=5)
#         probs = top_k_probs.numpy().tolist()
#         classes = top_k_indices.numpy().tolist()
#         classes = [n+1 for n in classes]
#         labels = [class_names[str(n+1)] for n in classes]

#         print(labels[np.argmax(probs)],max(probs))

if __name__ == '__main__':
    Main()
    
