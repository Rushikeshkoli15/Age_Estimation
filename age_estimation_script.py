import torch
import numpy as np
import os
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from PIL import Image
import torch
import shutil
from face_detector.models.experimental import attempt_load
from face_detector.detect_face import detect
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align
import time
import imghdr
from webptools import grant_permission
from webptools import dwebp
from PIL import Image, ImageSequence
import imagehash
import psycopg2
import schedule
import time, datetime
import requests
import json

grant_permission()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

# Check if GPU is available
cudadevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {cudadevice}")

starttime = time.time()

app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106'],
                   providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


classes=  {'0':'0_2', '1':'11_14', '2':'15_17', '3':'18_22', '4':'23_27', '5':'28_36', '6':'37_48', '7':'3_6', '8':'49_65', '9':'66_100', '10':'7_10'}


class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=11):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None

# define a transformation to resize the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_age_vit(face):
    #face=transform(face)
    # Apply feature extractor, stack back into 1 tensor and then convert to tensor
    images = torch.tensor(np.stack(feature_extractor(face)['pixel_values'], axis=0))
    images = images.to(cudadevice)
    # Feed through model
    outputs, _ = age_model(images, None)
    # Get predicted class
    predicted = torch.argmax(outputs, dim=1)
    age_pred_cls = classes[str(predicted.item())]
    #print("age **********", age_pred_cls)
    return age_pred_cls


def is_child(age_pred_cls):
  if age_pred_cls in  ['0_2', '3_6', '7_10', '11_14']:#, '15_17']:

    return True

#face_model = load_model('./face_detector/yolov5s-face.pt', device)
face_model = attempt_load('./face_detector/yolov5s-face.pt', map_location=cudadevice)  # load FP32 model

# Define Model
age_model = ViTForImageClassification()
# Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# Load trained model state dict
#age_model.load_state_dict(torch.load("model_weights.pt", map_location=torch.device('cpu')))
age_model.load_state_dict(torch.load("model_weights.pt", map_location=cudadevice))

# Set the model to eval mode
age_model.eval()
age_model.to(cudadevice)

def detect_csam(image_path):


  
  #Part1: detect face uing insightface
  try:
    img = cv2.imread(image_path)
    #imzsize= img.shape()
  except Exception as e:
    print("Image Not Proper:",e)
    #shutil.copyfile(image_path, os.path.join("test_images", "imread", im))
    return 'CORRUPT','303','False'
    
  try:
    faces = app.get(img)
  except Exception as e:
    print("insightface detector ERROR:",e)
    #shutil.copyfile(image_path, os.path.join("test_images", "insight", im))
    faces = []
    return 'CORRUPT','303','False'
    
     
    # Checking with IMAGE SIZE 112
    for i, face in enumerate(faces):
      face1 = face_align.norm_crop(img, image_size=112, landmark=faces[i].kps)      
      face = Image.fromarray(np.uint8(face1)).convert('RGB')
      age_pred_cls = get_age_vit(face)
      
      
      if is_child(age_pred_cls):
         print("The given image contains child")
         return "Child"
        
      else:
         print("The given image does not contain child")
         return "ADult"



      
      
    
