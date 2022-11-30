from keras import Model
from keras.saving.save import load_model

source_model_name = 'h5/resnet50.h5'


# conv5_block3_out layer before avg_pool
def getSourceModel(out_layer_name='avg_pool'):
    global source_model_name
    base_model = load_model(source_model_name)

    model = Model(inputs=base_model.input, outputs=base_model.get_layer(out_layer_name).output)

    return model


def freezeSource(model):
    for layer in model.layers:
        layer.trainable = False
    return model
