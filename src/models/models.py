from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
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
    X = Dense(fc0_nodes, activation='relu', activity_regularizer=l2(weight_decay), name='fc0')(X)
    X = Dense(1, bias_initializer=output_bias, name='logits')(X)
    Y = Activation('sigmoid', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    return model