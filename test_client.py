#!/usr/bin/env python

import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from gender_age_service.srv import *

def prediction_client(image_message):
    rospy.wait_for_service('prediction')
    try:
        prediction = rospy.ServiceProxy('prediction', GenderPrediction)
        reponsePrediction = prediction(image_message)
	print 'REPONSE : ', reponsePrediction
        return reponsePrediction
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    #creer image_message
    bridge = CvBridge()
    cv_image = cv2.imread("/home/quentin/catkin_ws/src/gender_age_service/src/image/olivier.jpg")
    #np_image_data = np.asarray(cv_image)
    image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

    prediction_client(image_message)
