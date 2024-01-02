from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import List
from pathlib import Path

from src.classifier_service import classify_image, classify_front_image, classify_smile_image, classify_sides_image

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
@app.post('/predict-frontal/')
async def classifier_endpoint_frontal(file_upload: UploadFile):
    try:
        data = await file_upload.read()
        save_to = UPLOAD_DIR / file_upload.filename
        with open (save_to, 'wb') as f:
            f.write(data)

        print('1')
        image_path = str(save_to)
        print(image_path)
        print('2')

        ## Change Type of Image HERE ##
        results = classify_front_image(image_path)
        
        print('3')

        return {
            'Frontal Results': results
        }

    except Exception as e:
        # Handle exceptions, return 500 Internal Server Error
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post('/predict-frontal-smile/')
async def classifier_endpoint_frontal(file_upload: UploadFile):
    try:
        data = await file_upload.read()
        save_to = UPLOAD_DIR / file_upload.filename
        with open (save_to, 'wb') as f:
            f.write(data)

        print('1')
        image_path = str(save_to)
        print(image_path)
        print('2')

        ## Change Type of Image HERE ##
        results = classify_smile_image(image_path)
        
        print('3')

        return {
            'Frontal-Smile Results': results
        }
    
    except Exception as e:
        # Handle exceptions, return 500 Internal Server Error
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post('/predict-sides/')
async def classifier_endpoint_side(file_upload: UploadFile):
    try:
        data = await file_upload.read()
        save_to = UPLOAD_DIR / file_upload.filename
        with open (save_to, 'wb') as f:
            f.write(data)

        print('1')
        image_path = str(save_to)
        print(image_path)
        print('2')

        ## Change Type of Image HERE ##
        results = classify_sides_image(image_path)
        
        print('3')

        return {
            'Sides Results': results
        }

    except Exception as e:
        # Handle exceptions, return 500 Internal Server Error
        return JSONResponse(content={"error": str(e)}, status_code=500)

### Classify Image - ORIGINAL ###
# @app.post('/predict/')
# async def classifier_endpoint(file_upload: UploadFile):
#     try:
#         data = await file_upload.read()
#         save_to = UPLOAD_DIR / file_upload.filename
#         with open (save_to, 'wb') as f:
#             f.write(data)

#         print('1')
#         image_path = str(save_to)
#         print(image_path)
#         print('2')

#         ## Change Type of Image HERE ##
#         results = classify_front_image(image_path)
        
#         print('3')

#         return {
#             'Results': results
#         }