#!/usr/bin/env python3
# Fridosleigh.com CAPTEHA API - Made by Krampus Hollyfeld
import requests
import json
import sys
import base64
from PIL import Image
from io import BytesIO

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import threading
import queue
import time

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def predict_image(q, sess, graph, image_bytes, img_full_path, labels, input_operation, output_operation):                                                                            
    image = read_tensor_from_image_bytes(image_bytes)                               
    results = sess.run(output_operation.outputs[0], {                               
        input_operation.outputs[0]: image                                           
    })                                                                            
    results = np.squeeze(results)                                                
    prediction = results.argsort()[-5:][::-1][0]                                 
    q.put( {'img_full_path':img_full_path, 'prediction':labels[prediction].title(), 'percent':results[prediction]} )                                                          
                                                                                       
def load_graph(model_file):                                                            
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def read_tensor_from_image_bytes(imagebytes, input_height=299, input_width=299, input_mean=0, input_std=255):
    image_reader = tf.image.decode_png( imagebytes, channels=3, name="png_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)
    return result


def main():
    yourREALemailAddress = "ENTER YOUR REAL EMAIL ADDRESS"

    # Creating a session to handle cookies
    s = requests.Session()
    url = "https://fridosleigh.com/"

    json_resp = json.loads(s.get("{}api/capteha/request".format(url)).text)
    b64_images = json_resp['images']                    # A list of dictionaries eaching containing the keys 'base64' and 'uuid'
    challenge_image_type = json_resp['select_type'].split(',')     # The Image types the CAPTEHA Challenge is looking for.
    print("WE ARE LOOKING FOR: " + challenge_image_type[0]+','+challenge_image_type[1]+','+challenge_image_type[2])
    challenge_image_types = [challenge_image_type[0].strip(), challenge_image_type[1].strip(), challenge_image_type[2].replace(' and ','').strip()] # cleaning and formatting
    
    '''

MISSING IMAGE PROCESSING AND ML IMAGE PREDICTION CODE GOES HERE


'''
    for image64 in b64_images:
        png = Image.open(BytesIO(base64.b64decode(image64['base64'])))
        png.save('unknown_images/'+image64['uuid'],'PNG')


    # Loading the Trained Machine Learning Model created from running retrain.py on the training_images directory
    graph = load_graph('/tmp/retrain_tmp/output_graph.pb')
    labels = load_labels("/tmp/retrain_tmp/output_labels.txt")

    # Load up our session
    input_operation = graph.get_operation_by_name("import/Placeholder")
    output_operation = graph.get_operation_by_name("import/final_result")
    sess = tf.compat.v1.Session(graph=graph)

    # Can use queues and threading to spead up the processing
    q = queue.Queue()
    unknown_images_dir = 'unknown_images'
    unknown_images = os.listdir(unknown_images_dir)
    
    #Going to interate over each of our images.
    for image in unknown_images:
        img_full_path = '{}/{}'.format(unknown_images_dir, image)
        
        #print('Processing Image {}'.format(img_full_path))
        # We don't want to process too many images at once. 10 threads max
        while len(threading.enumerate()) > 100:
            time.sleep(0.0001)

        #predict_image function is expecting png image bytes so we read image as 'rb' to get a bytes object
        image_bytes = open(img_full_path,'rb').read()
        threading.Thread(target=predict_image, args=(q, sess, graph, image_bytes, img_full_path, labels, input_operation, output_operation)).start()
    
    print('Waiting For Threads to Finish...')
    while q.qsize() < len(unknown_images):
        time.sleep(0.001)
    
    #getting a list of all threads returned results
    prediction_results = [q.get() for x in range(q.qsize())]
     
    #do something with our results... Like print them to the screen.
    guess = ""
    for prediction in prediction_results:
        #print('TensorFlow Predicted {img_full_path} is a {prediction} with {percent:.2%} Accuracy'.format(**prediction)) 
        #print("WE ARE LOOKING FOR: " + challenge_image_types[0])
        if prediction['prediction'] in challenge_image_types:
            #print('Matched: ' + prediction['img_full_path'])
            if guess == "":
                guess = prediction['img_full_path'].split('/')[1]
            else:
                guess = guess+','+prediction['img_full_path'].split('/')[1]
    print(guess)
## End of Paste


    # This should be JUST a csv list image uuids ML predicted to match the challenge_image_type .
    #final_answer = ','.join( [ img['uuid'] for img in b64_images ] )
    final_answer=guess
    json_resp = json.loads(s.post("{}api/capteha/submit".format(url), data={'answer':final_answer}).text)
    if not json_resp['request']:
        # If it fails just run again. ML might get one wrong occasionally
        print('FAILED MACHINE LEARNING GUESS')
        print('--------------------\nOur ML Guess:\n--------------------\n{}'.format(final_answer))
        print('--------------------\nServer Response:\n--------------------\n{}'.format(json_resp['data']))
        sys.exit(1)

    print('CAPTEHA Solved!')
    # If we get to here, we are successful and can submit a bunch of entries till we win
    userinfo = {
        'name':'Krampus Hollyfeld',
        'email':yourREALemailAddress,
        'age':180,
        'about':"Cause they're so flippin yummy!",
        'favorites':'thickmints'
    }
    # If we win the once-per minute drawing, it will tell us we were emailed. 
    # Should be no more than 200 times before we win. If more, somethings wrong.
    entry_response = ''
    entry_count = 1
    while yourREALemailAddress not in entry_response and entry_count < 200:
        print('Submitting lots of entries until we win the contest! Entry #{}'.format(entry_count))
        entry_response = s.post("{}api/entry".format(url), data=userinfo).text
        entry_count += 1
    print(entry_response)


if __name__ == "__main__":
    main()
