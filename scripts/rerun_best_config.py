import hpbandster.core.result as hpres
from vdrnn.vrnn import VRNN

import os
import json
import argparse

'''
This script re-run with a given dataset the best configuration found by BOHB for a certain budget and then saves the model in a folder.
'''

parser = argparse.ArgumentParser(description='''re-run for a given budget the best configuration (incumbent) found by BOHB with a given dataset.''')

parser.add_argument('--folder', type=str, help='name of the folder with the results', default='./')
parser.add_argument('--budget', type=float, help='time budget in minutes', default=10)
parser.add_argument('--dataset', type=str, help='dataset to be used for training', default='mnist')
parser.add_argument('--model_folder', type=str, help='name folder where the model is saved', default='model')

args=parser.parse_args()

# load the example run from the log files
result = hpres.logged_results_to_HBS_result(args.folder)

# get all executed runs
all_runs = result.get_all_runs()

# get the 'dict' that translates config ids to the actual configurations
id2conf = result.get_id2config_mapping()

# Here is how you get he incumbent (best configuration)
inc_id = result.get_incumbent_id()

# let's grab the run on the highest budget
inc_runs = result.get_runs_by_id(inc_id)
inc_run = inc_runs[-1]
inc_config = id2conf[inc_id]['config']

print('Best found configuration:')

print(inc_config)

#available dataset
available_datasets  = ['mnist', 'higgs', 'vehicle', 'adult']

if args.dataset not in available_datasets:
    raise Exception('{} is not available. Please choose among the available datasets {}'.format(args.dataset, available_datasets))

#extract the data
PATH = '/home/matilde/Documents/vdrnn/'

#data to be used: convnet_cifar10
with open(os.path.join(PATH, 'surrogate_data', '%s.json'%args.dataset),'r') as f:
    data = json.load(f)

lc = data['learning_curves']
theta = data['config_space']

print('Number of learning curves {}'.format(len(lc)))

#create a vrnn with the indicated dataset
net = VRNN(lc, train_config_data=theta)

final_lr = inc_config['final_learning_rate_fraction']*inc_config['initial_learning_rate']

#convert the configuration in the right format
config = {'act':'relu', 'batch_size':inc_config['batch_size'],
          'hs':inc_config['num_lstm_layers']*[inc_config['num_lstm_units_per_layer']],
          'mlp':inc_config['num_mlp_layers']*[inc_config['num_mlp_units_per_layer']],
          'mlp_config':inc_config['num_mlp_config']*[inc_config['num_mlp_config_units_per_layer']],
          'prob':1-inc_config['dropout_rate'],'lr':inc_config['initial_learning_rate'],
          'scheduler':inc_config['learning_rate_schedule'],
          'budget':args.budget*60, 'momentum': inc_config['momentum'],
          'final_lr':final_lr, 'max_cutting_length': inc_config['min_observed_epochs'],
          'curriculum_learning':'linear'}

#train the network for the given budget with the given configuration
res = net.train_vrnn(hyper_par=config, save=True, output_path=os.path.join(PATH, 'saved_models'), dir_name=args.model_folder)

#save the test and train losses
with open(os.path.join(PATH, 'saved_models', args.model_folder, 'losses.json'), 'w') as f:
    json.dump(res, f)
