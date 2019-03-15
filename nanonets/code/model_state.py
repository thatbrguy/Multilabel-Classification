import requests 
import json
import os

BASE_URL = 'http://app.nanonets.com/api/v2/MultiLabelClassification/'
AUTH_KEY = os.environ.get('NANONETS_API_KEY')
MODEL_ID = os.environ.get('NANONETS_MODEL_ID')

def get_model(model_id):
    url = BASE_URL + "Model/%s"%(model_id)
    response = requests.request('GET', url, auth=requests.auth.HTTPBasicAuth(AUTH_KEY,''))
    res =  json.loads(response.text)
    state, status  = res['state'], res['status']
    if state != 5:
        print("The model isn't ready yet, it's status is:", status)
        print("We will send you an email when the model is ready. If your imapatient, run this script again in 10 minutes to check.")
        print("\n\nmore details at:")
        print("https://app.nanonets.com/multilabelclassification/#/classify/"+model_id)
    else:
        print("NEXT RUN: python ./code/prediction.py ./multilabel_data/ImageSets/1067.jpg")


if __name__=="__main__":
    get_model(MODEL_ID)
