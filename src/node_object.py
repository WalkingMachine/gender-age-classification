#!/usr/bin/env python

import rospy

from threading import Thread
from time import sleep

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
from sara_msgs.msg import GenderAgePrediction

global new_picture
global new_response
global listPrediction
global image

# noinspection PyInterpreter
class NodePrediction(object):

    def graph_and_prediction(self):
        #variables declaree global pour partager avec le main
        global new_picture
        global new_response
        global listPrediction
        global image
        new_picture = False
        new_response = False
        listPrediction = None
        image = None
        ##creation du modele
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
        self.g1=Graph()
        print("GAPH IS READY")
        ##Fin intitialisation du graph

        #tant que pas shutdown = toujours vrai
        while not rospy.is_shutdown():
            ##detection de l'arrivee d'une photo
            if new_picture == True:
            ##classification lorsqu'il y a une photo
                listPrediction = prediction_age_gender.prediction(image, self.loaded_model, self.g1)
                new_picture = False
                new_response = True
            else:
                sleep(0.3)

    #lorsqu'une image arrive
    def handle_prediction(self, image_message):
        #variables declaree global pour partager avec le thread
        global new_picture
        global new_response
        global listPrediction
        global image
        new_picture = False
        new_response = False
        listPrediction = None
        ## arrivee de la photo et passage en global pour aller au thread
        image = image_message.image
        new_picture = True
        ## attend la reponse
        while new_response == False:
            sleep(0.1)
        ##recuperer la reponse
        # storePrediction = GenderAgePrediction()
        resp = ListGenderPredictionResponse()

        if listPrediction is not None:
            for myPrediction in listPrediction:
                storePrediction = GenderAgePrediction()
                ages=myPrediction[1].split(', ')
                storePrediction.ageMin = int(ages[0][1:len(ages[0])])
                storePrediction.ageMax = int(ages[1][0:len(ages[1])-1])
                storePrediction.probAge = myPrediction[2]
                storePrediction.probFemale = myPrediction[3][0]
                storePrediction.probMale = myPrediction[3][1]
                resp.listPrediction.append(storePrediction)
                print "store"
                print storePrediction
            print "resp"
            print resp

        new_response = False

        return resp

    #intitialisation
    def __init__(self):
        #service
        rospy.init_node('prediction_server')
        s = rospy.Service('prediction', ListGenderPrediction, self.handle_prediction)

        #variables pour detecter nouvelle image et fin classification
        new_picture = False
        new_response = False
        listPrediction = None

        #creation du thread avec le graph
        try:
           t = Thread(target=self.graph_and_prediction)
           t.start()
        except:
           print "Error: unable to start thread"
        print "Ready to predict gender."
        #spin
        rospy.spin()

    def __del__(self):
        self.g1.close_sess()
