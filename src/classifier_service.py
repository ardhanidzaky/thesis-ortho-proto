import torch

from src.utils import get_models, prep_image_for_inference, prep_image_for_inference_side
from const import *

MODELS_DICT = {
    'Front': {
        'Tipe': get_models('tipe', 3)
        , 'Simetris': get_models('simetris', 2)
        , 'Horizontal': get_models('horizontal', 2)
        , 'Vertikal': get_models('vertikal', 2)
    }

    , 'Smile': {
        'Segaris': get_models('segaris', 3)
        , 'Bukal': get_models('bukal', 3)
        , 'Kurva': get_models('kurva', 3)
        , 'Garis': get_models('garis', 3)
    }

    , 'Side': {
        'Profil': get_models('profil', 3)
        , 'Nasolabial': get_models('nasolabial', 3)
        , 'Mentolabial': get_models('mentolabial', 3)
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
    print('2a')
    image = prep_image_for_inference(image_path)
    print('2b')

    predicted_type = model_predict(MODELS_DICT['Front']['Tipe'], image)
    print('2c')
    predicted_symmetry = model_predict(MODELS_DICT['Front']['Simetris'], image)
    predicted_horizontal = model_predict(MODELS_DICT['Front']['Horizontal'], image)
    predicted_vertikal = model_predict(MODELS_DICT['Front']['Vertikal'], image)

    return {
        'Tipe Wajah': TIPE_WAJAH[predicted_type.item()]
        , 'Simetris Wajah': TIDAK_YA[predicted_symmetry.item()]
        , 'Keseimbangan Transversal': TIDAK_YA[predicted_horizontal.item()]
        , 'Keseimbangan Vertikal': TIDAK_YA[predicted_vertikal.item()]
    }

def classify_smile_image(image_path):
    print('2a')
    image = prep_image_for_inference(image_path)
    print('2b')

    predicted_segaris = model_predict(MODELS_DICT['Smile']['Segaris'], image)
    print('2c')
    predicted_bukal = model_predict(MODELS_DICT['Smile']['Bukal'], image)
    predicted_kurva = model_predict(MODELS_DICT['Smile']['Kurva'], image)
    predicted_garis = model_predict(MODELS_DICT['Smile']['Garis'], image)

    return {
        'Garis Midline Wajah': TIDAK_YA[predicted_segaris.item()]
        , 'Bukal Koridor': BUKAL_MULUT[predicted_bukal.item()]
        , 'Kurva Senyum': KURVA_MULUT[predicted_kurva.item()]
        , 'Garis Senyum': GARIS_MULUT[predicted_garis.item()]
    }

def classify_sides_image(image_path):
    print('2a')
    image = prep_image_for_inference_side(image_path)
    print('2b')

    predicted_profil = model_predict(MODELS_DICT['Side']['Profil'], image)
    print('2c')
    predicted_nasolabial = model_predict(MODELS_DICT['Side']['Nasolabial'], image)
    predicted_mentolabial = model_predict(MODELS_DICT['Side']['Mentolabial'], image)

    return {
        'Profil Wajah': PROFIL_WAJAH[predicted_profil.item()]
        , 'Sudut Mentolabial': MESO_NESO[predicted_mentolabial.item()]
        , 'Sudut Nasolabial': MESO_NESO[predicted_nasolabial.item()]
    }

def model_predict(model, image):
    print('2d')
    model.eval()
    print('2e')
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    
    return predicted

    
