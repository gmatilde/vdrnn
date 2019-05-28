from vdrnn.vrnn import VRNN

import os
import json

import argparse

'''
This script evaluates a model on the different dataset mnist, higgs, vehicle and adult.
'''

parser = argparse.ArgumentParser(description='''Evaluate the model on the different dataset mnist, higgs, vehicle and adult.''')

parser.add_argument('--model_folder', type=str, help='name folder where model is saved', default='model_mnist')

args=parser.parse_args()

#path to project
PATH = '/home/matilde/Documents/vdrnn/'

data_path = os.path.join(PATH, 'surrogate_data')

save_path = os.path.join(PATH, 'saved_models')

#available datasets
available_datasets  = ['mnist', 'higgs', 'vehicle', 'adult']

for dataset in available_datasets:

    print('\n\nevaluate model on {} dataset\n'.format(dataset))

    #data to be used: convnet_cifar10
    with open(os.path.join(data_path, '%s.json'%dataset),'r') as f:
        data = json.load(f)

    lc = data['learning_curves']
    theta = data['config_space']

    print('Number of learning curves {}'.format(len(lc)))

    #create a vrnn with the indicated dataset
    net = VRNN(lc, train_config_data=theta)

    with open(os.path.join(save_path, args.model_folder, 'hyper.json'), 'r') as f:
        hyper_par = json.load(f)
    f.close()

    with open(os.path.join(save_path, args.model_folder, 'par.json'), 'r') as f:
        par = json.load(f)
    f.close()

    net.load_model(hyper_par, par)

    res = net.eval(n_samples=3)

    #save results of evaluation
    with open(os.path.join(save_path, args.model_folder, '%s.json'%dataset), 'w') as f:
        json.dump(res, f)
    f.close()
