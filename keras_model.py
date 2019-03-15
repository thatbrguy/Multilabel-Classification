from keras.callbacks import Callback
from keras.backend import clear_session
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten
from keras.applications import ResNet50, MobileNet, Xception, DenseNet121

def build_model(mode, model_name = None, model_path = None):

    clear_session()

    if mode == 'train':
        img = Input(shape = (224, 224, 3))

        if model_name == 'DenseNet121':

            model = DenseNet121(include_top=False, 
                                weights='imagenet', 
                                input_tensor=img, 
                                input_shape=None, 
                                pooling='avg')

        elif model_name == 'MobileNet':

            model = MobileNet(include_top=False, 
                              weights='imagenet', 
                              input_tensor=img, 
                              input_shape=None, 
                              pooling='avg')

        elif model_name == 'Xception':

            model = Xception(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=None, 
                             pooling='avg')

        elif model_name == 'ResNet50':

            model = ResNet50(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=None, 
                             pooling='avg')

        final_layer = model.layers[-1].output

        dense_layer_1 = Dense(128, activation = 'relu')(final_layer)
        output_layer = Dense(10, activation = 'sigmoid')(dense_layer_1)

        model = Model(input = img, output = output_layer)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

    elif mode == 'inference':
        model = load_model(model_path)

    return model

