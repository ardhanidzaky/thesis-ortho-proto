from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import List
from pathlib import Path

from src.classifier_service import classify_image, classify_front_image

UPLOAD_DIR = Path() / 'uploads'

app = FastAPI()
app.add_middleware(
    CORSMiddleware
    , allow_origins=['*']
    , allow_credentials=True
    , allow_methods=['*']
    , allow_headers=['*']
)

### Health Check ###
@app.get('/')
async def root():
    return {
        'message': 'Hi Ralfi!'
    }

### Upload File Trial ###
@app.post('/upload/')
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    save_to = UPLOAD_DIR / file_upload.filename
    with open (save_to, 'wb') as f:
        f.write(data)
    
    return {'Filename': file_upload.filename}

### Classify Image ###
### 1 image only ###
@app.post('/predict/')
async def classifier_endpoint(file_upload: UploadFile):
    """
    Classify subtask for a front side orthodontic image.

    Parameters:
    - file: uploaded image.

    Returns:
    - A dictionary containing the classification results for each type of subtask.

    Raises:
    - 500 Internal Server Error
    """
    try:
        # Save image uploaded by users.
        data = await file_upload.read()
        save_to = UPLOAD_DIR / file_upload.filename
        with open (save_to, 'wb') as f:
            f.write(data)

        image_path = str(save_to)
        results = classify_front_image(image_path)

        return {
            'Results': results
        }

    except Exception as e:
        # Handle exceptions, return 500 Internal Server Error
        return JSONResponse(content={"error": str(e)}, status_code=500)

### [TODO] More than 1 image ###
# @app.post('/predict/')
# async def classifier_endpoint(file_upload: List[UploadFile]):
#     """
#     Classify three different types of images: front, smile, and sides.

#     Parameters:
#     - files (List[UploadFile]): List of three uploaded images.

#     Returns:
#     - A dictionary containing the classification results for each type of image.

#     Raises:
#     - 500 Internal Server Error
#     """
#     try:
#         results_front = classify_image(files[0], type='front')
#         results_smile = classify_image(files[1], type='smile')
#         results_sides = classify_image(files[2], type='sides')

#         return {
#             'Front Image Results': results_front
#             , 'Smile Image Results': results_smile
#             , 'Sides Image Results': results_sides
#         }
#     except Exception as e:
#         # Handle exceptions, return 500 Internal Server Error
#         return JSONResponse(content={"error": str(e)}, status_code=500)