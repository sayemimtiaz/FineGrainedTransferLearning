from datetime import datetime
from models.cifar100.data_util import getFineGrainedClass
from keras.models import load_model
from util.transfer_util import train


target_names = ['target_3_0_0.01','traditional_transfer', 'no_transfer']
# target_names = ['traditional_transfer', 'no_transfer']

x_train, y_train, x_test, y_test, num_classes = getFineGrainedClass(superclass='vehicles 1')

for target_name in target_names:
    print('Evaluating ' + target_name)

    target_name = 'transfer_model/' + target_name + '.h5'
    model = load_model(target_name)

    # print(model.summary())
    print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

    train(model, x_train, y_train, x_test, y_test, epochs=30)
