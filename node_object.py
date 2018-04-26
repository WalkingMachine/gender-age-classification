#!/usr/bin/env python

import rospy

import prediction_age_gender
from gender_age_service.srv import *

import rospkg
from keras.models import load_model
from keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt
from build_predicator import *
from utils import *
from cv_bridge import CvBridge, CvBridgeError

# noinspection PyInterpreter
class NodePrediction(object):

    def handle_prediction(self, image_message):

        if self.counter == 0:
            # # une seule fois
            rospack = rospkg.RosPack()
            path = rospack.get_path('gender_age_service') + "/src/freeze_graph/model.json"
            json_file = open(path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.loaded_model = model_from_json(loaded_model_json, custom_objects={"tf": tf})
            # # load weights into new model
            path = rospack.get_path('gender_age_service') + "/src/freeze_graph/model.h5"
            self.loaded_model.load_weights(path)
            print("Loaded model from disk")
            resp = None
            self.counter = 1

        self.g1 = Graph()
        listPrediction = prediction_age_gender.prediction(image_message, self.loaded_model, self.g1)

        resp = GenderPredictionResponse()
        resp.ageRange = listPrediction[0][1]
        resp.probAge = listPrediction[0][2]
        resp.probFemale = listPrediction[0][3][0]
        resp.probMale = listPrediction[0][3][1]
        return resp

    def __init__(self):
        rospy.init_node('prediction_server')
        s = rospy.Service('prediction', GenderPrediction, self.handle_prediction)
        self.counter = 0
        print "Ready to predict gender."
        rospy.spin()

    def __del__(self):
        self.g1.close_sess()