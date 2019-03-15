import requests 
import json
import os
import sys


BASE_URL = 'http://app.nanonets.com/api/v2/MultiLabelClassification/'
AUTH_KEY = os.environ.get('NANONETS_API_KEY')
MODEL_ID = os.environ.get('NANONETS_MODEL_ID')


def predict(image_file_path, model_id):
    url = BASE_URL + 'Model/%s/LabelFiles/'%(model_id)
    data = {'files': open(image_file_path, 'rb')}

    response = requests.post(url, auth= requests.auth.HTTPBasicAuth(AUTH_KEY, ''), files=data)
    print(json.loads(response.text))

if __name__=="__main__":
    image_file_path = sys.argv[1]
    predict(image_file_path, MODEL_ID)
