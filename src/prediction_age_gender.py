from keras.models import load_model
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from utils import *
import cv2
from cv_bridge import CvBridge, CvBridgeError
from build_predicator import *
from utils import *
import matplotlib.pyplot as plt
import rospkg

def prediction(image, loaded_model, g1):
    input_size = 416
    max_box_per_image = 10
    anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    labels = ["face"]

    # DEBUT ANALYSE DE L'IMAGE
    #de l'image message au format lisible pour le prog
    bridge = CvBridge()
    image_cv = bridge.imgmsg_to_cv2(image)
    image2 = cv2.resize(image_cv, (input_size, input_size))
    image2 = normalize(image2)
    input_image = image2[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    dummy_array = np.zeros((1, 1, 1, 1, max_box_per_image, 4))
    input_data = [input_image, dummy_array]

    #detection des visages
    netout = loaded_model.predict([input_image, dummy_array])[0]
    boxes = decode_netout2(netout, labels, anchors)

    #si au moins un visage est detecte
    if len(boxes) != 0:
        listImg = getFacesList(image_cv, boxes)
        #classification homme-femme et age
        listPrediction = g1.classify_age(listImg)
    #sinon retourne vide
    else:
        listPrediction = None

    print listPrediction
    return listPrediction
