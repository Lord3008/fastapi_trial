import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from BankNotes import BankNote
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify domains instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
# @app.get('/{name}')
# def get_name(name: str):
#     return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload