import PIL
from PIL import Image

import torch as tr
import torch.nn as nn

import torchvision
import torchvision.transforms as T
import torchvision.models as models

import cv2

def prep_image_for_inference(image_path):
    data_transform = return_data_transform((600, 800))
    rgb_image, _ = crop_face(image_path, target_width=600, target_height=800)
    rgb_image = Image.fromarray(rgb_image)
    rgb_transformed = data_transform(rgb_image).unsqueeze(0)

    return rgb_transformed

def crop_face(image_path, target_width=300, target_height=400):
    """
    Crop and resize the detected face in an image.

    Parameters:
    - image_path (str): The path to the input image file.
    - target_width (int, optional): The desired width of the output face image (default is 300 pixels).
    - target_height (int, optional): The desired height of the output face image (default is 400 pixels).

    Returns:
    - resized_rgb (numpy.ndarray): The cropped and resized face region as a NumPy array in Gray color format.
    - resized_gray (numpy.ndarray): The cropped and resized face region as a NumPy array in Gray color format.

    Raises:
    - ValueError: If the image cannot be loaded from the given path or if no faces are detected in the image.

    This function takes an image file, detects the face in the image, and resizes it to match the specified target 
    width and height while maintaining the aspect ratio. The result is returned as a NumPy array in RGB and Gray color format.

    Example usage:
    >>> original_image, gray_iamge = crop_face("your_image.jpg")
    """

    image = cv2.imread(image_path)
    image_rb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Search for faces on the image.
    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    x, y, w, h = faces[0]

    # Adjust height, so it will create a 3:4 (width:height) ratio.
    exp_ratio = 3 / 4
    h = int(w / exp_ratio)

    # Adjust y, as a pre-caution if it 
    # being cropped below the forehead.
    y -= int((image.shape[0] / target_height) * 35)
    
    # Add padding for the height, as a pre-caution
    # if it being cropped below the forehead.
    if y + h > image.shape[0]:
        minus_y = y + h - image.shape[0]
        y -= minus_y

    image_cropped = image_rb[y:y+h, x:x+w]
    image_cropped_resized = cv2.resize(image_cropped, (target_width, target_height))
    
    resized_rgb = image_cropped_resized
    resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2GRAY)

    return resized_rgb, resized_gray

def load_model_checkpoint(model, ckp_path):
    """
    Load a PyTorch model from a checkpoint file.
    This function loads a pre-trained or saved PyTorch 
    model from a checkpoint file and returns the loaded model.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be loaded.
    - ckp_path (str): The path to the checkpoint file.

    Returns:
    - loaded_model (torch.nn.Module): The model loaded from the checkpoint.

    Example Usage:
    >>> # Load a pre-trained model from a checkpoint file
    >>> model = load_model_checkpoint(models.resnet18(pretrained=False), 'model_checkpoint.pth')

    Note:
    - Ensure that the model architecture in the checkpoint file matches the provided `model` argument.
    - Make sure to import the necessary PyTorch model modules from `torchvision.models`.
    """
    temp_model = model
    checkpoint = tr.load(ckp_path)
    temp_model.load_state_dict(checkpoint)

    return temp_model

def get_models(subtask, output_size):
    """
    Get a dictionary of pre-trained models.
    This function loads and returns a dictionary of pre-trained PyTorch models, 
    including EfficientNet B0, ShuffleNet V2, and MobileNet V2.

    Parameters:
    - subtask (str): Subtask that want to be predicted.
    - output_size (int): How many class this model predicts.

    Returns:
    - model_dict (dict): A dictionary with model names as keys and the corresponding pre-trained models as values.

    Example Usage:
    >>> # Get a dictionary of pre-trained models
    >>> models_dict = get_models()
    >>> effnet_model = models_dict['effnet']
    >>> shufflenet_model = models_dict['shufflenet']
    >>> mobilenet_model = models_dict['mobilenet']

    Note:
    - Make sure to import the necessary PyTorch model modules from `torchvision.models`.
    """
    efficientnet_b0 = load_model_checkpoint(load_efficient_net(output_size), f'assets/{subtask}_effnet.pth')  
    shufflenet = load_model_checkpoint(load_shuffle_net(output_size), f'assets/{subtask}_shufflenet.pth')
    mobilenet_v2 = load_model_checkpoint(load_mobile_net(output_size), f'assets/{subtask}_mobilenet.pth')

    model_dict = {}
    model_dict['effnet'] = efficientnet_b0
    model_dict['shufflenet'] = shufflenet
    model_dict['mobilenet'] = mobilenet_v2

    return model_dict

def load_efficient_net(output_size):
    temp_model = models.efficientnet_b0(pretrained=False)

    model = nn.Sequential(*list(temp_model.children())[:-1])
    model.add_module('global_avg_pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc', nn.Linear(temp_model.classifier[-1].in_features, output_size))

    return model

def load_mobile_net(output_size):
    temp_model = models.mobilenet_v2(pretrained=False)

    model = nn.Sequential(*list(temp_model.children())[:-1])
    model.add_module('global_avg_pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc', nn.Linear(temp_model.classifier[-1].in_features, output_size))

    return model

def load_shuffle_net(output_size):
    temp_model = models.shufflenet_v2_x1_0(pretrained=False)

    model = nn.Sequential(*list(temp_model.children())[:-1])
    model.add_module('global_avg_pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc', nn.Linear(temp_model.fc.in_features, output_size))

    return model

def return_data_transform(desired_size):
    data_transform = T.Compose([
        T.Resize(desired_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use the ImageNet mean and std
    ])

    return data_transform