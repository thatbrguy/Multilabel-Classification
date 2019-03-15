# Multilabel-Classification
Repository containing code for the blog post titled "Working Title"

## Using Keras

## Method 1: Google Colab
- You can explore this notebook on Colab to directly experiment with training the models.

## Method 2: Local Setup

### Step 1: Clone the Repo
```bash
git clone https://github.com/thatbrguy/Multilabel-Classification.git
cd Multilabel-Classification
```

### Step 2: Download the Dataset
  - Download [data.tar.gz](https://drive.google.com/open?id=1Kuz9LVt9nxFghTwDeo9csu0lnNdIbmu8) and place it in the current directory.
  - Extract the dataset using `tar -xzvf data.tar.gz`
  - Move the contents of `./data/keras/` to the current directory by using `mv ./data/keras/* ./`
  - Run `python extract_data.py`

## Step 3: Train the Model
  - Run `train.py` to train the model.

## Using Nanonets
 
### Step 1: Clone the Repo
```bash
git clone https://github.com/thatbrguy/Multilabel-Classification.git
cd Multilabel-Classification
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
The training data is found in ```images``` (image files) and ```annotations``` (annotations for the image files)
```bash
python ./code/upload_training.py
```

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
python ./code/prediction.py ./multilabel_data/ImageSets/2795.jpg
```

## References
1. Recipes5k
2. Nanonets
