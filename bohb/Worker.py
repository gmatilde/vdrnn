import json
import os
import numpy as np

import ConfigSpace as CS
from hpbandster.core.worker import Worker

from vdrnn.vrnn import VRNN

class VRNN_worker(Worker):

	def __init__(self, file_data, path='./', **kwargs):

		# call super with all the hpbandster related keyword arguments
		super().__init__(**kwargs)

		path = os.path.join(path, file_data)

		with open(path,'r') as f:
			data = json.load(f)

		self.lc = data['learning_curves']
		self.theta = data['config_space']

	def compute(self, config, budget, config_id,**kwargs):

		instance = VRNN(self.lc, train_config_data=self.theta)

		hyper_par = self.translate_config(config, budget)

		#train my network
		losses = instance.train_vrnn(hyper_par)

		#evaluate the model
		res = instance.eval(n_samples=3)

		post_ll = np.asarray(res['test log_likelihood'])

		return_dict = {
			'loss': -np.mean(post_ll),
			'info': {
				'mean_ll': np.mean(post_ll),
				'std_ll': np.std(post_ll),
				'median_ll': np.median(post_ll),
				'iqr_ll': np.percentile(post_ll, 75) - np.percentile(post_ll, 25),
				'mse_loss': losses['loss'],
				'mse_test_loss': losses['test loss']
			}
		}

		return(return_dict)

	@staticmethod
	def translate_config(hpb_config, budget):

		vdrnn_config = {
			'hs': [hpb_config['num_lstm_units_per_layer']]*hpb_config['num_lstm_layers'],
			'mlp':[hpb_config['num_mlp_units_per_layer']]*hpb_config['num_mlp_layers'],
			'mlp_config':[hpb_config['num_mlp_config_units_per_layer']]*hpb_config['num_mlp_config'],
			'act': 'relu',
			'max_cutting_length': hpb_config['min_observed_epochs'],
			'budget' : budget,
			'batch_size': hpb_config['batch_size'],
			'momentum': hpb_config['momentum'],
			'prob': 1-hpb_config['dropout_rate'],
			'lr': hpb_config['initial_learning_rate'],
			'scheduler': hpb_config['learning_rate_schedule'],
			'curriculum_learning':'linear',
			'TF': True,
			'final_lr' : hpb_config['initial_learning_rate']*hpb_config['final_learning_rate_fraction']
		}

		return(vdrnn_config)

	@staticmethod
	def get_configspace():

		config_space = CS.ConfigurationSpace()

		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('batch_size', lower=4, upper=128, log=True, default_value=32))
		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('min_observed_epochs', lower=5, upper=50, log=True, default_value=10))
		config_space.add_hyperparameter(CS.UniformFloatHyperparameter('momentum', lower=0, upper=0.99, log=False, default_value=0.99))
		config_space.add_hyperparameter(CS.UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.99, log=False, default_value=0.1))
		config_space.add_hyperparameter(CS.UniformFloatHyperparameter('initial_learning_rate', lower=1e-5, upper=1e-1, log=True, default_value=1e-3))
		config_space.add_hyperparameter(CS.UniformFloatHyperparameter('final_learning_rate_fraction', lower=1e-4, upper=1e-0, log=True, default_value=1e-2))
		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_lstm_layers', lower=1, upper=2, log=False, default_value=1))
		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_mlp_layers', lower=1, upper=2, log=False, default_value=1))
		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_mlp_config', lower=1, upper=2, log=False, default_value=1))
		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_lstm_units_per_layer', lower=4, upper=64, log=True, default_value=16))
		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_mlp_units_per_layer', lower=4, upper=128, log=True, default_value=16))
		config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_mlp_config_units_per_layer', lower=4, upper=128, log=True, default_value=16))
		config_space.add_hyperparameter(CS.CategoricalHyperparameter('learning_rate_schedule', ['const','exp', 'cos']))

		return(config_space)

if __name__ == '__main__':

	worker = VRNN_worker('mnist.json', path='/home/matilde/Documents/vdrnn/surrogate_data', run_id='test')
	configspace = worker.get_configspace()

	while(True):
		config = configspace.sample_configuration()
		print(worker.compute(config=config, budget=1, config_id=(0,0,0)))
