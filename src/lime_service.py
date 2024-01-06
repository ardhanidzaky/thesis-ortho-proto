import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from src.utils import get_image_frontal, get_image_sides, get_pil_transform, get_preprocess_transform

from lime import lime_image
from skimage.segmentation import mark_boundaries

resize = get_pil_transform()
preprocess = get_preprocess_transform() 

def lime_for_frontal(model, image_path, file_naming):
    img = get_image_frontal(image_path)
    model = model
    print('Lime 2')

    def batch_predict(image):
        model.eval()

        batch = torch.stack(tuple(preprocess(i) for i in image), dim=0)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        
        return probs.detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(resize(img)),
        batch_predict, # classification function
        top_labels=25, 
        hide_color=0, 
        num_samples=250 # number of images that will be sent to classification function
    )
    print('Lime 3')

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.imsave(f'lime_result/{image_path.split("/")[1]}_{file_naming}.png', img_boundry1) 
    print('Lime 4')

    return 'Success'

def lime_for_sides(model, image_path, file_naming):
    img = get_image_sides(image_path)
    model = model
    print('Lime 2')

    def batch_predict(image):
        model.eval()

        batch = torch.stack(tuple(preprocess(i) for i in image), dim=0)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        
        return probs.detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(resize(img)),
        batch_predict, # classification function
        top_labels=25, 
        hide_color=0, 
        num_samples=250 # number of images that will be sent to classification function
    )
    print('Lime 3')

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.imsave(f'lime_result/{image_path.split("/")[1]}_{file_naming}.png', img_boundry1) 

    print('Lime 4')
    
    return 'Success'