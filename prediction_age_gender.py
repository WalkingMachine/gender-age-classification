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

def prediction(image_message):

	input_size=416
	max_box_per_image   = 10
	anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
	labels=["face"]
	#image_path   = "image/olivier.jpg"

	## load json and create model
	rospack = rospkg.RosPack()
	path = rospack.get_path('gender_age_service')+"/src/freeze_graph/model.json"
	print 'PATH', path
	json_file = open(path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json, custom_objects={"tf": tf})
	## load weights into new model
	path = rospack.get_path('gender_age_service')+"/src/freeze_graph/model.h5"
	print 'PATH', path
	loaded_model.load_weights(path)
	print("Loaded model from disk")

	#global g1

	#loaded_model._make_predict_function()
	g1=Graph()
	

	#DEBUT ANALYSE DE L'IMAGE
		
	#image = cv2.imread(image_path)

	#with sess.graph.as_default():
	bridge = CvBridge()
	image = bridge.imgmsg_to_cv2(image_message.image)
	image2 = cv2.resize(image, (input_size, input_size))

	image2 = normalize(image2)

	input_image = image2[:,:,::-1]
	input_image = np.expand_dims(input_image, 0)
	dummy_array = dummy_array = np.zeros((1,1,1,1,max_box_per_image,4))
	input_data=[input_image, dummy_array]
	netout = loaded_model.predict([input_image, dummy_array])[0]
	boxes  = decode_netout2(netout, labels,anchors)
		
	if len(boxes) != 0:
	    listImg=getFacesList(image, boxes)
	    listPrediction=g1.classify_age(listImg)
	    #image = draw_boxes_v2(image, boxes, labels,listPrediction)
	    #print(len(boxes), 'boxes are found')
	    print(listPrediction)
	    #imgplot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	    #plt.show()
	    #cv2.imwrite(image_path[:-4] + '_detected2' + image_path[-4:], image)
	
	print listPrediction
	return listPrediction

