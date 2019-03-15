import requests 
import json
import os
import random

BASE_URL = 'http://app.nanonets.com/api/v2/MultiLabelClassification/'
AUTH_KEY = os.environ.get('NANONETS_API_KEY')
MODEL_ID = os.environ.get('NANONETS_MODEL_ID')

image_folder_path = "./multilabel_data/ImageSets/"
annotations_folder = "./multilabel_data/Annotations/"

def create_image_dictionary():
    image_label_dictionary = {}
    for image in os.listdir(image_folder_path):
        annotatios_file_name  = os.path.join(annotations_folder, "%s.txt"%(image.rsplit('.', 1)[0]))
        if not os.path.isfile(annotatios_file_name):
            continue
        all_labels = [x for x in open(annotatios_file_name, 'r').read().split('\n') if x]
        image_label_dictionary[os.path.join(image_folder_path, image)] = all_labels
    return image_label_dictionary

def upload_images(model_id):
    url = BASE_URL + 'Model/%s/UploadFiles/'%(model_id)
    print ("Uploading Images......")
    image_label_dictionary = create_image_dictionary()
    all_images = list(image_label_dictionary.keys())
    random.shuffle(all_images)
    n = len(all_images)
    image_uploaded = 0
    while len(all_images) > 0:
        batch_images, all_images = all_images[:50], all_images[50:]
        multiple_files = []
        multiple_data = []
        for image in batch_images:
            labels = image_label_dictionary[image]
            image_dir, image_name = os.path.split(image)
            image_data = {'filename': image_name, "categories": labels}
            multiple_data.append(image_data)
            multiple_files.append(('files', (image_name, open(image, 'rb'), 'image/jpeg')))
        multiple_files.append(('data', ('', json.dumps(multiple_data))))
        response = requests.post(url, auth= requests.auth.HTTPBasicAuth(AUTH_KEY, ''), files=multiple_files)
        image_uploaded += len(batch_images)
        if len(all_images) > 0:
            print ("%d of %d images has been uploaded, uploading next batch...."%(image_uploaded, n))
        else:
            print ("%d of %d images has been uploaded, Done uploading"%(image_uploaded, n))
    return 

if __name__=="__main__":
    upload_images(MODEL_ID)
    print("\n\n\nNEXT RUN: python ./code/train_model.py")