# Multilabel-Classification
Repository containing Keras code for the blog post titled "How to Perform Multi Label Classification using Deep Learning". You can checkout the blog post [here](#).

<p align="center">
  <img src="output_sample.png" alt="Harry Potter and the Golden Snitch" height="301px" width="750px"></img>
</p>

## Using Keras
This section lists out the steps involved in training a Keras model (with TensorFlow backend) for Multi Label Classification.

## Method 1: Google Colab
- You can explore this notebook on [Colab](https://colab.research.google.com/drive/1OdZYPxQm4e_y4lpsDkKy_MwhrEpYXooZ#offline=true&sandboxMode=true) to directly experiment with training the models.

## Method 2: Local Setup
Follow these steps to train and use a model for Multilabel Classification. You can also directly use a sample trained model  (`mobilenet.h5`) without training, which can be downloaded from [here](https://drive.google.com/open?id=1K2-eqcoBEJURHJ0K4FoCF70Ei6YYUcNs) (skip to Step 4 in that case). 

### Step 1: Clone the Repo
```bash
git clone https://github.com/thatbrguy/Multilabel-Classification.git
cd Multilabel-Classification
```

### Step 2: Download the Dataset
  - Download [data.tar.gz](https://drive.google.com/open?id=1hKFMGIY2jNYbntK4e4aGvwOwvzptP8DH) and place it in the current directory.
  - Extract the dataset using `tar -xzvf data.tar.gz`
  - Move the contents of `./data/keras/` to the current directory by using `mv ./data/keras/* ./`

## Step 3: Train the Model
  - Run `train.py --model ResNet50` to train the model.
  - The `--model` argument can take one among `ResNet50`, `MobileNet`, `DenseNet121` or `Xception`.

## Step 4: Inference
  - Run `predict.py --image PATH_TO_FILE --saved_model PATH_TO_h5` to obtain a prediction once the model is trained. 
  - `PATH_TO_FILE` refers to the path of the image.
  - `PATH_TO_h5` refers to the path of the h5 file.

## Using Nanonets
This section lists out the steps involved in training a Nanonets model for Multi Label Classification.

### Step 1: Clone the Repo
```bash
git clone https://github.com/thatbrguy/Multilabel-Classification.git
cd Multilabel-Classification/nanonets
```

### Step 2: Get your free API Key
Get your free API Key from http://app.nanonets.com/user/api_key

### Step 3: Set the API key as an Environment Variable
```bash
export NANONETS_API_KEY=YOUR_API_KEY_GOES_HERE
```

### Step 4: Create a New Model
```bash
python ./code/create_model.py
```
 >_**Note:** This generates a MODEL_ID that you need for the next step

### Step 5: Add Model Id as Environment Variable
```bash
export NANONETS_MODEL_ID=YOUR_MODEL_ID
```
 >_**Note:** you will get YOUR_MODEL_ID from the previous step

### Step 6: Upload the Training Data
  - Download [data.tar.gz](https://drive.google.com/open?id=1hKFMGIY2jNYbntK4e4aGvwOwvzptP8DH) and place it in the current directory.
  - Extract the dataset using `tar -xzvf data.tar.gz`
  - Move the contents of `./data/nanonets/` to the current directory by using `mv ./data/nanonets/* ./`
  - Run `python ./code/upload_training.py` to upload the data.

### Step 7: Train Model
Once the Images have been uploaded, begin training the Model
```bash
python ./code/train_model.py
```

### Step 8: Get Model State
The model takes ~2 hours to train. You will get an email once the model is trained. In the meanwhile you check the state of the model
```bash
python ./code/model_state.py
```

### Step 9: Make Prediction
Once the model is trained. You can make predictions using the model
```bash
python ./code/prediction.py PATH_TO_YOUR_IMAGE.jpg
```

**Sample Usage:**
```bash
python ./code/prediction.py ./multilabel_data/ImageSets/2_my_caesar_salad_hostedLargeUrl.jpg
```

## References
1. [Recipes5k](http://www.ub.edu/cvub/recipes5k/)
2. [Nanonets](https://nanonets.com/)
