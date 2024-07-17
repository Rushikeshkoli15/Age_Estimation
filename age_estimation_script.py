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


def is_CSA(age_pred_cls):
  if age_pred_cls in  ['0_2', '3_6', '7_10', '11_14']:#, '15_17']:

    return True


def hamming2(s1, s2):
    """
    provides the hamming distance between two hex strings.
    """
    # convert hex strings to binary
    s1 = bin(int('1' + s1, 16))[3:]
    s2 = bin(int('1' + s2, 16))[3:]
    # Calculate the Hamming distance between two bit strings
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    
def check_zero_hash(phash):
  if phash == "0000000000000000" or phash == "8000000000000000":
    return True
  elif hamming2("0000000000000000", phash) < 4 or hamming2("8000000000000000", phash) < 4:
    return True
  else:
    return False


#face_model = load_model('./face_detector/yolov5s-face.pt', device)
face_model = attempt_load('./face_detector/yolov5s-face.pt', map_location=cudadevice)  # load FP32 model

# Define Model
age_model = ViTForImageClassification()
# Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# Load trained model state dict
#age_model.load_state_dict(torch.load("/data/raid/ML/pooja/VIT/saved_age_model_20april/final_model.pt", map_location=torch.device('cpu')))
age_model.load_state_dict(torch.load("/data/raid/ML/pooja/VIT/saved_age_model_20april/final_model.pt", map_location=cudadevice))

# Set the model to eval mode
age_model.eval()
age_model.to(cudadevice)

def detect_csam(image_path):
  csam_flag = False
  pjs_flag = False
  #print(image_path)
  
  final_bb_app = []
  final_landmark_app = []
  final_face_age_app = {}
  final_bb_yolo = []
  final_landmark_yolo = []
  final_face_age_yolo = {}
  dict1,dict2,face_results,final_dict = {},{},{},{}
  
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
    
    
  if len(faces) > 0: 
    for i, face in enumerate(faces):
      final_bb_app = []
      final_landmark_app = []
      final_bb_app.append(face['bbox'])
      final_bb_app1 = [arr.tolist() for arr in final_bb_app]
      #final_bb_app1 = final_bb_app1[0]
      final_bb_app1 = [round(num, 2) for sublist in final_bb_app1 for num in sublist]
      final_landmark_app.append(face['kps']) 
      final_landmark_app1 = [arr.tolist() for arr in final_landmark_app]
      #final_landmark_app1 = final_landmark_app1[0]
      final_landmark_app1 = [[round(num, 2) for num in sublist] for sublist in final_landmark_app1[0]]
      final_face_age_app[i] = []
      
      dict1[i] = {'bb':final_bb_app1,'landmark':final_landmark_app1,'expressions':'-20'}
    
    
    # Checking with IMAGE SIZE 112
    for i, face in enumerate(faces):
      face1 = face_align.norm_crop(img, image_size=112, landmark=faces[i].kps)      
      face = Image.fromarray(np.uint8(face1)).convert('RGB')
      age_pred_cls = get_age_vit(face)
      #save_path = os.path.join(face_save_path,  str(i) + "__" + im) 
      #im1 = face1.save(save_path)
      
      x1 = final_face_age_app[i]
      x1.append(age_pred_cls)
      final_face_age_app[i]=x1
      dict1[i]['age']=x1
      
      if is_CSA(age_pred_cls):
        csam_flag = True
        #cv2.imwrite(save_path, face1)
        # break
        
      if csam_flag == False and age_pred_cls=='15_17':
        pjs_flag = True
      
      
    face_results['buffalo'] = dict1
    face_results['yolo'] = 'NA'
  
  elif len(faces) == 0:
    face_results['buffalo'] = 'NA'
    face_results['yolo'] = 'NA'



  
  
  final_dict['face_results'] = face_results

  
  return csam_flag,final_dict,pjs_flag


def detect_csam_in_gifwebp(image_path):
  csam_flag = False
  index = 0
  MediaFile = Image.open(image_path)
  for frame in ImageSequence.Iterator(MediaFile):
    if frame.mode != 'RGB':
      frame = frame.convert('RGB')
    
    phash = str(imagehash.phash(frame))  # Phash value for Image
    frame_save_path = "test_images/webpjpg/" + str(index) + "__" + im
    if index == 0 or hamming2(phash_prev, phash) > 4:
      frame.save(frame_save_path)
      if detect_csam(frame_save_path):
        return csam_flag, frame_save_path
        
    phash_prev =  phash            
    index += 1


  return csam_flag, None      





dir1 = "path to image dir"
images = os.listdir(dir1)

print("RECEIVED IMAGE COUNT :",len(images))
    
if len(images) > 0:
  for im in images:
    image_path = '/data'+im
    print(image_path)
    
    if not os.path.exists(image_path):
      print("IMAGE not Found")
      continue

    
    csam_flag,ca_info2,pjs_flag = detect_csam(image_path)
    
    
    # IMAGE IS CSAM
    if csam_flag == True: 
      print("CSAM")   
      
    
    # IMAGE IS PORN
    elif csam_flag == False and pjs_flag == False:
      print("PORN")
      
      
    # IMAGE IS PJS
    elif csam_flag == False and pjs_flag == True:
      print("PJS")

    
    # IMAGE IS CORRUPT
    elif csam_flag == 'CORRUPT':
      print("CORRUPT")
      
      
      
      
