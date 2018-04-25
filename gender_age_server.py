#!/usr/bin/env python

from gender_age_service.srv import *
import rospy

import prediction_age_gender

import rospkg
from keras.models import load_model
from keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt
from build_predicator import *
from utils import *

def handle_prediction(image_message):
    listPrediction = prediction_age_gender.prediction(image_message)
    resp = GenderPredictionResponse()
    resp.ageRange = listPrediction[0][1]
    resp.probAge = listPrediction[0][2]
    resp.probFemale = listPrediction[0][3][0]
    resp.probMale = listPrediction[0][3][1]
    return resp

def prediction_server():
    # load json and create model
    #rospack = rospkg.RosPack()
    #path = rospack.get_path('gender_age_service')+"/src/freeze_graph/model.json"
    #json_file = open(path, 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #global loaded_model
    #loaded_model = model_from_json(loaded_model_json, custom_objects={"tf": tf})
    # load weights into new model
    #path = rospack.get_path('gender_age_service')+"/src/freeze_graph/model.h5"
    #print 'PATH', path
    #loaded_model.load_weights(path)

    #print("Loaded model from disk")

    #global g1
    #g1=Graph()
    
    rospy.init_node('prediction_server')
    s = rospy.Service('prediction', GenderPrediction, handle_prediction)
    print "Ready to predict gender."
    rospy.spin()

if __name__ == "__main__":
    prediction_server()


