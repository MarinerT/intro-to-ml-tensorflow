#!/usr/bin/python3

import argparse
import json as simplejson
from utils import *
from PIL import Image

def Main():
    #parse out variables
    parser = argparse.ArgumentParser(description='Image Classifier.')

    parser = argparse.ArgumentParser()
    #mandatory arguments 
    parser.add_argument('path', help='string; filepath of image')
    parser.add_argument('model', help='.h5 file')
    
    #not mandatory arguments
    parser.add_argument('--top_k', help='integer; the number of top responses', action='store_true',default=5)
    parser.add_argument('--category_names', help='a json file; map of label to catetgory',action='store_true',default='./label_map.json')

    args = parser.parse_args()

    #creating the model
    # this version makes the keras version so we can use dot notation.
    model = tf.keras.models.load_model(args.model)


    #map labels
    with open(args.category_names,'r') as f:
        class_names = simplejson.load(f)

    #make predictions
    probs, classes = predict(args.path, model, args.top_k)
    classes = classes.numpy().tolist()
    labels = class_names.get(str(classes))

    #outputs

    #print the top_k and their associated probabilities
    if args.top_k:
        for _ in range(args.top_k):
            print('\t\u2022' + probs[_] + ':' + labels[_])
            
    #print the most likely label & it's associated probability
    else:
        print(labels[np.argmax(probs)],max(probs))

if __name__ == '__main__':
    Main()
    
