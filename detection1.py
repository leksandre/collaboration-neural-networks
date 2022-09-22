import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from imageai.Detection import ObjectDetection
from datetime import datetime
os.chdir('C:\python_work08052021') 

# from keras.models import load_model
# base_model=load_model('C:\python_work08052021\model_ex-003_acc-0.657895.h5')
# base_model.save('C:\python_work08052021\model_ex-003_acc-0.657895.h5_111111111111111111.h5', include_optimizer=False)



# from keras.models import load_model
# from keras.applications.vgg16 import VGG16
# # model = VGG16(weights = 'imagenet')
# model = VGG16(weights = None)
# model.load_weights('model_ex-001_acc-0.842105.h5')
# model = load_model('model_ex-001_acc-0.842105.h5')
# from keras.applications.vgg16 import VGG16
# # model = VGG16(weights = 'model_ex-001_acc-0.842105.h5',input_shape=5)
# model = VGG16(weights = 'imagenet')




execution_path = os.getcwd()
now = datetime.now()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()


detector.setModelPath( os.path.join(execution_path , 'resnet50_coco_best_v2.1.0.h5'))#model_ex-003_acc-0.718750.h5
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
# detector.setModelPath( os.path.join(execution_path , "DenseNet-BC-121-32.h5"))
# detector.setModelPath( os.path.join(execution_path , "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
# detector.setModelPath( os.path.join(execution_path , "resnet50_imagenet_tf.2.0.h5"))
# detector.setModelPath( os.path.join(execution_path , "mobilenet_v2.h5"))
detector.loadModel()

# detections = detector.detectObjectsFromImage(
#     input_image=os.path.join(execution_path , "image.jpg"), 
#     output_image_path=os.path.join(execution_path , "image_new"+now.strftime("%m_%d_%Y_%H_%M_%S")+".jpg"),
#     minimum_percentage_probability=1,
# 	display_percentage_probability=True,
# 	display_object_name=True
#     )

# custom = detector.CustomObjects(person=True, dog=True,   train=True)
custom = detector.CustomObjects(car=True, motorcycle=True,  bus=True,   truck=True)
detections = detector.detectObjectsFromImage( 
    # custom_objects=custom, 
    input_image=os.path.join(execution_path , "C:\python_work08052021\image000028.jpg"), 
    output_image_path=os.path.join(execution_path , "image_new"+now.strftime("%m_%d_%Y_%H_%M_%S")+".jpg"),
    minimum_percentage_probability=40
    #,display_object_name=True
    )

# # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))


# detections = detector.detectObjectsFromImage(input_image="C:\python_work08052021\image411.jpg", output_image_path="C:\python_work08052021\imagenew411_21.jpg", minimum_percentage_probability=30)


# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"]
    
    
     )
print("--------------------------------")