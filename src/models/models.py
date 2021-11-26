from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant

def get_model(model_name):
    '''
    Return the model definition and associated preprocessing function as specified in the config file
    :return: (TF model definition function, preprocessing function)
    '''

    if model_name == 'mobilenetv2':
        model_def = mobilenetv2
        preprocessing_function = mobilenetv2_preprocess
    elif model_name == 'inceptionv3':
        model_def = inception_v3
        preprocessing_function = inceptionv3_preprocess
    elif model_name == 'cutoffvgg16':
        model_def = CutoffVGG16
        preprocessing_function = vgg16_preprocess
    else:
        raise Exception("Invalid entry in TRAIN > MODEL_DEF field of config.yml.")
    return model_def, preprocessing_function


def mobilenetv2(model_config, input_shape, metrics, n_classes, output_bias=None):
    '''
    Defines a model based on a pretrained MobileNetV2 for binary US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)
    weight_decay = model_config['L2_LAMBDA']
    fc0_nodes = model_config['NODES_FC0']
    frozen_layers = model_config['FROZEN_LAYERS']
    print("MODEL HYPERPARAMETERS: ", model_config)

    if output_bias is not None:
        output_bias = Constant(output_bias)  # Set initial output bias

    # Start with pretrained MobileNetV2
    X_input = Input(input_shape, name='input')
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input,
                             alpha=0.75)

    # Freeze layers
    for layers in range(len(frozen_layers)):
        layer2freeze = frozen_layers[layers]
        print('Freezing layer: ' + str(layer2freeze))
        base_model.layers[layer2freeze].trainable = False

    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    #X = Dense(fc0_nodes, activation='relu', activity_regularizer=l2(weight_decay), name='fc0')(X)
    X = Dense(1, name='logits')(X)
    Y = Activation('sigmoid', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

def inception_v3(model_config, input_shape, metrics, n_classes, output_bias=None):
    '''
        Defines a model based on a pretrained InceptionV3 for binary US classification.
        :param model_config: A dictionary of parameters associated with the model architecture
        :param input_shape: The shape of the model input
        :param metrics: Metrics to track model's performance
        :param output_bias: bias initializer of output layer
        :return: a Keras Model object with the architecture defined in this method
        '''

    # Set hyperparameters
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)
    weight_decay = model_config['L2_LAMBDA']
    fc0_nodes = model_config['NODES_FC0']
    frozen_layers = model_config['FROZEN_LAYERS']
    print("MODEL HYPERPARAMETERS: ", model_config)

    if output_bias is not None:
        output_bias = Constant(output_bias)  # Set initial output bias

    # Start with pretrained INceptionV3
    X_input = Input(input_shape, name='input')
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    # Freeze layers
    for layers in range(len(frozen_layers)):
        layer2freeze = frozen_layers[layers]
        print('Freezing layer: ' + str(layer2freeze))
        base_model.layers[layer2freeze].trainable = False

    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    #X = Dense(fc0_nodes, activation='relu', activity_regularizer=l2(weight_decay), name='fc0')(X)
    X = Dense(1, bias_initializer=output_bias, name='logits')(X)
    Y = Activation('sigmoid', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

class CutoffVGG16:

    def __init__(self, model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
        self.lr_extract = model_config['LR_EXTRACT']
        self.lr_finetune = model_config['LR_FINETUNE']
        self.dropout = model_config['DROPOUT']
        self.cutoff_layer = model_config['CUTOFF_LAYER']
        self.finetune_layer = model_config['FINETUNE_LAYER']
        self.extract_epochs = model_config['EXTRACT_EPOCHS']
        self.optimizer_extract = Adam(learning_rate=self.lr_extract)
        self.optimizer_finetune = RMSprop(learning_rate=self.lr_finetune)
        self.output_bias = output_bias
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.metrics = metrics
        self.mixed_precision = mixed_precision
        self.model = self.define_model()

    def define_model(self):
        X_input = Input(shape=self.input_shape, name='input')
        vgg16 = VGG16(input_shape=self.input_shape, include_top=False, weights='imagenet')
        self.vgg16_layers = vgg16.layers[1:self.cutoff_layer]
        X = X_input
        for layer in self.vgg16_layers:
            X = layer(X)
        X = GlobalAveragePooling2D(name='global_avgpool')(X)
        X = Dropout(self.dropout)(X)
        Y = Dense(1, activation='sigmoid', name='output')(X)
        model = Model(inputs=X_input, outputs=Y)
        model.summary()
        return model

    def fit(self, train_data, steps_per_epoch=None, epochs=1, validation_data=None, validation_steps=None,
            callbacks=None, verbose=1, class_weight=None):
        for layer in self.vgg16_layers:
            layer.trainable = False
        self.model.compile(optimizer=self.optimizer_extract, loss='binary_crossentropy', metrics=self.metrics, run_eagerly=True)
        history_extract = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=self.extract_epochs,
                            validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks,
                            verbose=verbose, class_weight=class_weight)
        for layer in self.vgg16_layers[self.finetune_layer:]:
            layer.trainable = True
        self.model.compile(optimizer=self.optimizer_finetune, loss='binary_crossentropy', metrics=self.metrics, run_eagerly=True)
        history_finetune = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=history_extract.epoch[-1],
                                      validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks,
                                      verbose=verbose, class_weight=class_weight)

    def evaluate(self, test_data, verbose=1):
        return self.model.evaluate(test_data, verbose=verbose)

    def predict(self, test_data, verbose=1):
        return self.model.predict(test_data, verbose=verbose)

    @property
    def metrics_names(self):
        return self.model.metrics_names