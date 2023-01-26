pretrained_architecures = ['resnet50', 'inceptionv3', 'vgg16', 'mobilenet']

SHAPE = (224, 224, 3)
# target_dataset = 'bird'
target_dataset = 'dog'
# target_dataset = 'cifar100'
# source_model_name = 'resnet50'
source_model_name = 'inceptionv3'
# source_model_name = 'inceptionresnetv2'

# source distribution related
source_dataset = 'tiny'
source_sample_size = 100
CLASS_WISE = False
