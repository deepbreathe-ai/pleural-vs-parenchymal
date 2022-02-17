from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_50v2_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, GlobalAveragePooling2D, BatchNormalization, \
    AveragePooling2D, Flatten, Conv2D, DepthwiseConv2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant

import tensorflow as tf

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
    elif model_name == 'resnet50v2':
        model_def = resnet50v2
        preprocessing_function = resnet_50v2_preprocess
    elif model_name == 'efficientnetb0':
        model_def = efficientnet
        preprocessing_function = efficientnet_preprocess
    elif model_name == 'resnet14v2':
        model_def = resnet14v2
        preprocessing_function = resnet_50v2_preprocess
    elif model_name == 'cutoffresnet50v2':
        model_def = cutoff_resnet50_v2
        preprocessing_function = resnet_50v2_preprocess
    elif model_name == 'vgg16':
        model_def = vgg16
        preprocessing_function = vgg16_preprocess
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

def resnet50v2(model_config, input_shape, metrics, n_classes, output_bias=None):
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
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

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

def efficientnet(model_config, input_shape, metrics, n_classes, output_bias=None):
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
    l2_regularizer = tf.keras.regularizers.l2(model_config['L2_LAMBDA'])
    fc0_nodes = model_config['NODES_FC0']
    frozen_layers = model_config['FROZEN_LAYERS']
    print("MODEL HYPERPARAMETERS: ", model_config)

    if output_bias is not None:
        output_bias = Constant(output_bias)  # Set initial output bias

    # Start with pretrained MobileNetV2
    X_input = Input(input_shape, name='input')
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    for layer_idx in range(len(base_model.layers)):
        layer = base_model.layers[layer_idx]

        # Freeze weights if necessary
        if layer_idx < frozen_layers:
            layer.trainable = False

        # Add L2 regularization to convolutional layers
        # if isinstance(layer, Conv2D) or isinstance(layer, DepthwiseConv2D):
        #     setattr(layer, 'activity_regularizer', l2_regularizer)

    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(fc0_nodes, activation='relu', name='fc0')(X)
    X = Dense(1, name='logits')(X)
    Y = Activation('sigmoid', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

def resnet14v2(model_config, input_shape, metrics, n_classes, output_bias=None):
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
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    # Freeze layers
    for layers in range(len(frozen_layers)):
        layer2freeze = frozen_layers[layers]
        print('Freezing layer: ' + str(layer2freeze))
        base_model.layers[layer2freeze].trainable = False

    #X = base_model.get_layer('conv2_block1_0_conv').output
    #X = base_model.get_layer('conv3_block4_3_conv').output
    X = base_model.get_layer('conv3_block4_out').output
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

def vgg16(model_config, input_shape, metrics, n_classes, output_bias=None):
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
    frozen_layers = model_config['FROZEN_LAYERS']
    print("MODEL HYPERPARAMETERS: ", model_config)

    if output_bias is not None:
        output_bias = Constant(output_bias)  # Set initial output bias

    # Start with pretrained MobileNetV2
    X_input = Input(input_shape, name='input')
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

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

class CutoffResnet50V2_old:

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
        rn50 = ResNet50V2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        self.rn50_layers = rn50.layers[1:self.cutoff_layer]
        X = X_input
        for layer in self.rn50_layers:
            print(layer(X))
            X = layer(X)
        X = GlobalAveragePooling2D(name='global_avgpool')(X)
        X = Dropout(self.dropout)(X)
        Y = Dense(1, activation='sigmoid', name='output')(X)
        model = Model(inputs=X_input, outputs=Y)
        model.summary()
        return model

    def fit(self, train_data, steps_per_epoch=None, epochs=1, validation_data=None, validation_steps=None,
            callbacks=None, verbose=1, class_weight=None):
        for layer in self.rn50_layers:
            layer.trainable = False
        self.model.compile(optimizer=self.optimizer_extract, loss='binary_crossentropy', metrics=self.metrics, run_eagerly=True)
        history_extract = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=self.extract_epochs,
                            validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks,
                            verbose=verbose, class_weight=class_weight)
        for layer in self.rn50_layers[self.finetune_layer:]:
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

def residual_block(X, num_filters: int, stride: int = 1, kernel_size: int = 3,
                   activation: str = 'relu', bn: bool = True, conv_first: bool = True, dropout: float = 0.2):
    """
    Parameters
    ----------
    X : Tensor layer - Input tensor from previous layer
    num_filters : int - Conv2d number of filters
    stride : int by default 1 - Stride square dimension
    kernel_size : int by default 3 - COnv2D square kernel dimensions
    activation: str by default 'relu' - Activation function to used
    bn: bool by default True - To use BatchNormalization
    conv_first : bool by default True - conv-bn-activation (True) or bn-activation-conv (False)
    """
    conv_layer = Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding='same',
                        kernel_regularizer=l2(1e-4))
    # X = input
    if conv_first:
        X = conv_layer(X)
        if bn:
            X = BatchNormalization()(X)
        if activation is not None:
            X = Activation(activation)(X)
            X = Dropout(dropout)(X)
    else:
        if bn:
            X = BatchNormalization()(X)
        if activation is not None:
            X = Activation(activation)(X)
        X = conv_layer(X)

    return X

def cutoff_resnet50_v2(model_config, input_shape, metrics, n_classes, output_bias=None):
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)
    frozen_layers = model_config['FROZEN_LAYERS']

    # depth should be 9n+2 (eg 56 or 110)
    # Model definition
    num_filters_in = 32
    num_res_block = int((model_config['DEPTH'] - 2) / 9)

    inputs = Input(shape=input_shape)

    # ResNet V2 performs Conv2D on X before spiting into two path
    X = residual_block(X=inputs, num_filters=num_filters_in, conv_first=True, dropout = dropout)

    # Building stack of residual units
    for stage in range(3):
        for unit_res_block in range(num_res_block):
            activation = 'relu'
            bn = True
            stride = 1
            # First layer and first stage
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if unit_res_block == 0:
                    activation = None
                    bn = False
                # First layer but not first stage
            else:
                num_filters_out = num_filters_in * 2
                if unit_res_block == 0:
                    stride = 2

            # bottleneck residual unit
            y = residual_block(X,
                               num_filters=num_filters_in,
                               kernel_size=1,
                               stride=stride,
                               activation=activation,
                               bn=bn,
                               conv_first=False)
            y = residual_block(y,
                               num_filters=num_filters_in,
                               conv_first=False)
            y = residual_block(y,
                               num_filters=num_filters_out,
                               kernel_size=1,
                               conv_first=False)
            if unit_res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                X = residual_block(X=X,
                                   num_filters=num_filters_out,
                                   kernel_size=1,
                                   stride=stride,
                                   activation=None,
                                   bn=False)
            X = tf.keras.layers.add([X, y])
        num_filters_in = num_filters_out


    # Add classifier on top.
    # v2 has BN-ReLU before Pooling

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = AveragePooling2D(pool_size=8)(X)

    # remove from initial code
    # y = Flatten()(X)
    # y = Dense(512, activation='relu')(y)
    # y = BatchNormalization()(y)
    # y = Dropout(dropout)(y)

    X = Dense(1, name='logits')(X)
    y = Activation('sigmoid', dtype='float32', name='output')(X)

    # remove from initial code
    # outputs = Dense(1, activation='softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=y)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
