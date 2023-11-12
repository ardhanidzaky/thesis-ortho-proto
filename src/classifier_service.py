import torch

from src.utils import get_models, prep_image_for_inference
from const import *

MODELS_DICT = {
    'Front': {
        'Tipe': get_models('tipe', 3)
        , 'Simetris': get_models('simetri', 2)
        , 'Horizontal': get_models('horizontal', 2)
        , 'Vertikal': get_models('vertikal', 2)
    }

    , 'Smile': {
        'None': None
    }

    , 'Side': {
        'None': None
    }
}

def classify_image(image_path, image_type):
    # <TODO: Write codes to direct image to each model.>
    model = None

    try:
        if image_type == 'front':
            return classify_front_image()
        elif image_type == 'smile':
            return classify_smile_image()
        else:
            return classify_sides_image()
    except:
        return 'Wrong image type!'

def classify_front_image(image_path):
    image = prep_image_for_inference(image_path)

    predicted_type = model_predict(MODELS_DICT['Front']['Tipe']['effnet'], image)
    predicted_symmetry = model_predict(MODELS_DICT['Front']['Simetris']['effnet'], image)
    predicted_horizontal = model_predict(MODELS_DICT['Front']['Horizontal']['effnet'], image)
    predicted_vertikal = model_predict(MODELS_DICT['Front']['Vertikal']['effnet'], image)

    return {
        'Tipe Wajah': TIPE_WAJAH[predicted_type.item()]
        , 'Simetris Wajah': TIDAK_YA[predicted_symmetry.item()]
        , 'Horizontal Transversal': TIDAK_YA[predicted_horizontal.item()]
        , 'Vertikal Transversal': TIDAK_YA[predicted_vertikal.item()]
    }

def classify_smile_image():
    return 'Halo'

def classify_sides_image():
    return 'Nice'

def model_predict(model, image):
    model.eval()
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    
    return predicted

    
