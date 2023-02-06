pretrained_architecures = [ 'xception', 'inceptionv3', 'densenet201']

target_datasets=['dog', 'bird']
# target_datasets=['bird']

SHAPE = (224, 224, 3)
# target_dataset = 'bird'
target_dataset = 'dog'
# target_dataset = 'cifar100'
# source_model_name = 'resnet50'
source_model_name = 'densenet201'
# source_model_name = 'inceptionresnetv2'

# source distribution related
# source_dataset = 'tiny'
source_dataset = 'imagenet'
CLASS_WISE = False
NUM_SOURCE_SAMPLE=10000


NUM_CLASSIFIER=1
DONE={}
CURRENT_ACQUIRE={}
