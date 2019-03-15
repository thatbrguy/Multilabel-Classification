import requests 
import json
import os

BASE_URL = 'http://app.nanonets.com/api/v2/MultiLabelClassification/'
AUTH_KEY = os.environ.get('NANONETS_API_KEY')
MODEL_ID = os.environ.get('NANONETS_MODEL_ID')

def train(model_id):
    url = BASE_URL + 'Model/%s/Train/'%(model_id)
    response = requests.request('POST', url, auth=requests.auth.HTTPBasicAuth(AUTH_KEY, ''))
    
    result = json.loads(response.text)
    print(result)

if __name__=="__main__":
    train(MODEL_ID)
    print("\n\nNEXT RUN: python ./code/model_state.py")
