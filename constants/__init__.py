pretrained_architecures = ['inceptionv3']

target_datasets=['dog', 'bird']
# target_datasets = ['pet', 'cats_vs_dogs', 'stl10']

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
CURRENT_ACQUIRE = {}
