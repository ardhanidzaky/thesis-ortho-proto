import PIL
from PIL import Image

import torch as tr
import torch.nn as nn

import torchvision
import torchvision.transforms as T
import torchvision.models as models

import cv2

import dlib

#####===== Image Preparation =====#####

def prep_image_for_inference(image_path):
    data_transform = return_data_transform((400, 400))
    rgb_image, _ = crop_face(image_path, 400, 400)
    rgb_image = Image.fromarray(rgb_image)
    rgb_transformed = data_transform(rgb_image).unsqueeze(0)

    return rgb_transformed

def prep_image_for_inference_side(image_path):
    data_transform = return_data_transform((400, 400))
    rgb_image = crop_face_side(image_path, 400, 400)
    rgb_image = Image.fromarray(rgb_image)
    rgb_transformed = data_transform(rgb_image).unsqueeze(0)

    return rgb_transformed

def crop_face(image_path, width, height):
    image = cv2.imread(image_path)
    image_rb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    x, y, w, h = faces[0]

    if width != height:
        exp_ratio = 3 / 4
        h = int(w / exp_ratio)
        y -= int((image.shape[0] / height) * 35)
        
        if y + h > image.shape[0]:
            minus_y = y + h - image.shape[0]
            y -= minus_y

        image_cropped = image_rb[y:y+h, x:x+w]
        image_cropped_resized = cv2.resize(image_cropped, (width, height))
        
        resized_rgb = image_cropped_resized
        resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2GRAY)
    else:
        image_cropped = image_rb[y:y+h, x:x+w]
        image_cropped_resized = cv2.resize(image_cropped, (width, height))
        
        resized_rgb = image_cropped_resized
        resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2GRAY)

    return resized_rgb, resized_gray

def crop_face_side(image_path, width, height):
    image = cv2.imread(image_path)
    image_fl = cv2.flip(image, 1)

    # For faster inference, resize to the largest 3:4 ratio we will use
    image_fl = cv2.resize(image_fl, (600, 800))

    detector = dlib.cnn_face_detection_model_v1("assets/mmod_human_face_detector.dat")
    faces = detector(image_fl)

    # If used
    desired_aspect_ratio = 3 / 4

    box_enlarge_factor = 1.5
    face = faces[0]

    center_x = int((face.rect.left() + face.rect.right()) / 2)
    center_y = int((face.rect.top() + face.rect.bottom()) / 2)

    if width != height:
        temp_width = int((face.rect.right() - face.rect.left()) * box_enlarge_factor)
        temp_height = int(temp_width / desired_aspect_ratio)
    else:
        temp_width = int((face.rect.right() - face.rect.left()) * box_enlarge_factor)
        temp_height = int((face.rect.bottom() - face.rect.top()) * box_enlarge_factor)

    # Calculate new coordinates for the square bounding box
    x1 = max(0, center_x - temp_width // 2)
    y1 = max(0, center_y - temp_height // 2)
    x2 = min(600, center_x + temp_width // 2)
    y2 = min(800, center_y + temp_height // 2)
        
    new_image = image_fl[y1:y2, x1:x2]
    new_image = cv2.resize(new_image, (width, height))

    return new_image

def return_data_transform(desired_size):
    data_transform = T.Compose([
        T.Resize(desired_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use the ImageNet mean and std
    ])

    return data_transform

#####===== Model Preparation =====#####

def load_model_checkpoint(model, ckp_path):
    temp_model = model
    checkpoint = tr.load(ckp_path, map_location=tr.device('cpu'))
    temp_model.load_state_dict(checkpoint)

    return temp_model

def get_models(subtask, output_size):
    shufflenet = load_model_checkpoint(load_shuffle_net(output_size), f'assets/{subtask}.pth')

    return shufflenet

def load_shuffle_net(output_size):
    temp_model = models.shufflenet_v2_x1_0(pretrained=False)

    model = nn.Sequential(*list(temp_model.children())[:-1])
    model.add_module('global_avg_pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc', nn.Linear(temp_model.fc.in_features, output_size))

    return model