import torch

from src.utils import get_models, prep_image_for_inference, prep_image_for_inference_side
from const import *

from src.lime_service import lime_for_sides, lime_for_frontal

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

def classify_front_image(image_path):
    print('2a')
    image = prep_image_for_inference(image_path)
    print('2b')

    predicted_type, lime_result1 = model_predict(MODELS_DICT['Front']['Tipe'], image, 'front', image_path, 'Tipe')
    print('2c')
    predicted_symmetry, lime_result2 = model_predict(MODELS_DICT['Front']['Simetris'], image, 'front', image_path, 'Simetris')
    predicted_horizontal, lime_result3 = model_predict(MODELS_DICT['Front']['Horizontal'], image, 'front', image_path, 'Horizontal')
    predicted_vertikal, lime_result4 = model_predict(MODELS_DICT['Front']['Vertikal'], image, 'front', image_path, 'Vertikal')

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

    predicted_segaris, lime_result1 = model_predict(MODELS_DICT['Smile']['Segaris'], image, 'front', image_path, 'Segaris')
    print('2c')
    predicted_bukal, lime_result2 = model_predict(MODELS_DICT['Smile']['Bukal'], image, 'front', image_path, 'Bukal')
    print('2ca')
    predicted_kurva, lime_result3 = model_predict(MODELS_DICT['Smile']['Kurva'], image, 'front', image_path, 'Kurva')
    print('2cb')
    predicted_garis, lime_result4 = model_predict(MODELS_DICT['Smile']['Garis'], image, 'front', image_path, 'Garis')
    print('2cd')

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

    predicted_profil, lime_result1 = model_predict(MODELS_DICT['Side']['Profil'], image, 'side', image_path, 'Profil')
    print('2c')
    predicted_nasolabial, lime_result2 = model_predict(MODELS_DICT['Side']['Nasolabial'], image, 'side', image_path, 'Mentolabial')
    predicted_mentolabial, lime_result3 = model_predict(MODELS_DICT['Side']['Mentolabial'], image, 'side', image_path, 'Nasolabial')

    return {
        'Profil Wajah': PROFIL_WAJAH[predicted_profil.item()]
        , 'Sudut Mentolabial': MESO_NESO[predicted_mentolabial.item()]
        , 'Sudut Nasolabial': MESO_NESO[predicted_nasolabial.item()]
    }

def model_predict(model, image, angle, image_path, file_naming):
    print('2d')
    model.eval()
    print('2e')
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    # Call LIME
    if angle == 'front':
        print('Lime 1')
        lime_result = lime_for_frontal(model, image_path, file_naming)
    else:
        print('Lime 1')
        lime_result = lime_for_sides(model, image_path, file_naming)
    
    return predicted, lime_result

    
