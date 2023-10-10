pretrained_architecures = ['xception', 'inceptionv3']
# pretrained_architecures = [  'vgg16']

target_datasets=['dog', 'bird', 'pet']
# target_datasets = ['dog', 'bird', 'stl10']

SHAPE = (224, 224, 3)
target_dataset = None
source_model_name = None

# source distribution related
# source_dataset = 'tiny'
source_dataset = 'imagenet'
CLASS_WISE = False
NUM_SOURCE_SAMPLE = 10000

NUM_CLASSIFIER = 1
DONE = {}

CURRENT_ACQUIRE = {'xception': 'bird'}


# DONE = {'inceptionv3': {'bird': ['pool']}, 'xception': {'bird': ['pool']}, 'vgg16': {'bird': ['pool']}}
