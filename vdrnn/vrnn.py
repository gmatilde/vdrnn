import json
import os

import numpy as np
import random
import time

import scipy.stats

import torch
import torch.nn as nn

import torch.distributions.bernoulli as Bernoulli

import sys
sys.path.append('..')

from vdrnn.model_lstm import LSTM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on {}'.format(device))

dtype = torch.float

class VRNN(object):
    """
    This class allows to create an instantion of variational LSTMs with user-defined
    architecture. In addition, it allows to train the model and evaluate its posterior.

    Inputs:

        training_data: dictionary containing the learning curves for the training.

        test_data: dictionary containing the learning curves for the test.

        train_config_data: dictionary containing the configuration vectors which originated the different learning curves in the training set.

        test_config_data: dictionary containing the configuration vectors which originated the different learning curves in the test set.
    """

    def __init__(self, train_data, test_data=None, train_config_data=None, test_config_data=None):
        """
        load the data and execute preprocessing (padding)
        """

        if test_data is not None and train_config_data is not None and test_config_data is None:
            raise Exception('Please provide configuration data also for the test set.')
        if test_data is not None and test_config_data is not None and train_config_data is None:
            raise Exception('Please provide configuration data also for the training set.')

        train_data, train_config_data = self.__pre_process(train_data, train_config_data)

        if test_data is None:

            train_data, train_config_data, test_data, test_config_data = self.__split_data(train_data, train_config_data)

        else:

            test_data, test_config_data = self.__pre_process(test_data, test_config_data)

        # note: training data dimensions: [time step, sample index, feature index]
        train_data = np.transpose([[datum for datum in train_data.values()]])

        test_data = np.transpose([[datum for datum in test_data.values()]])

        self.train_data_t = torch.tensor(train_data, device=device, dtype=dtype)

        self.test_data_t = torch.tensor(test_data, device=device, dtype=dtype)

        train_config_data = [datum for datum in train_config_data.values()]
        self.train_config_data_t = torch.tensor(train_config_data, device=device, dtype=dtype)

        test_config_data = [datum for datum in test_config_data.values()]
        self.test_config_data_t = torch.tensor(test_config_data, device=device, dtype=dtype)

        #size of configuration vector
        if self.train_config_data_t.shape[0]>0:
            self.config_size = self.train_config_data_t.shape[1]
        else:
            self.config_size = None

        self.lstm = None

        print('N_train = %i\nN_test=%i'%(self.train_data_t.shape[1], self.test_data_t.shape[1]))

    def __split_data(self, data, config=None):
        '''
        Internal use only:

        A method that splits the data in train and test
        '''

        indices = list(data.keys())

        random.seed(100)
        random.shuffle(indices)

        test_indices = indices[0:len(indices)//4]
        train_indices = indices[len(indices)//4:]

        #re-order the learning curves
        test_data = {}
        test_config = {}
        train_data = {}
        train_config = {}

        for i, ii in enumerate(train_indices):
            train_data[i] = data[ii]
            if config is not None and len(config)>0:
                train_config[i] = config[ii]

        for i, ii in enumerate(test_indices):
            test_data[i] = data[ii]
            if config is not None and len(config)>0:
                test_config[i] = config[ii]

        return train_data, train_config, test_data, test_config

    def __pre_process(self, data, config=None):
        '''
        Internal use only:

        A function that preprocess the data by padding the shorter learning curves
        with the padding value np.nan
        '''

        #first extract the maximum length of the learning curves
        max_length = max([len(lc) for lc in data.values()])

        data_new = {}
        config_new = {}

        for idx, datum in enumerate(data.values()):
            padded_datum = datum.copy()
            if (len(datum)<max_length):
                padded_datum.extend(np.nan*np.ones(max_length-len(datum)))
            data_new[idx] = padded_datum
            if config is not None:
                config_new[idx] = config[str(idx)]

        return data_new, config_new

    def __check_hyper(self, hyper_par, save, output_path, dir_name):
        """
        Internal use only:

        A function that checks the consistency of the hyperparameters.
        """

        # a valid default config with values for all parameters (even inactive ones)
        config = {
            'hs': [128],
            'mlp': [128],
            'mlp_config': [32],
            'batch_size': 32,
            'budget': 180, #budget in seconds
            'prob':  0.9,
            'act': 'relu',
            'lr': 0.01,
            'momentum': 0,
            'scheduler': 'const',
            'num_lr_steps': 2,
            'final_lr': 0,
            'curriculum_learning': 'linear',
            'max_cutting_length': self.train_data_t.shape[0],
            'cl_every_epoch': 1,
            'deactivate_TF': 10**10
        }

        activation_mapping = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }

        list_scheduler = ['const', 'step', 'exp', 'cos']

        cl_states = ['rand', 'seq', 'const', 'linear']

        # fill in values from hyper_par into default config, so every value is set
        config.update(hyper_par)

        #check values for hyperparameters
        try:
            for x in config['hs']:
                if not(isinstance(x,int)) or x<=0:
                    raise Exception('The number of hidden units per hidden layers for the stacked LSTMs has to be a positive integer.')
        except:
            raise ValueError('The definition of the recurrent layers must be an iterable with integers, e.g. [64, 64] for a two layer stacked LSTM with 64 units in each layer.')

        try:
            for x in config['mlp']:
                if not(isinstance(x,int)) or x<=0:
                    raise Exception('The number of hidden units per hidden layers for the mlp final layers has to be a positive integer.')
        except:
            raise ValueError('The definition of the feed forward layers must be an iterable with integers, e.g. [64, 64] for a two layer MLP with 64 units in each layer.')

        try:
            for x in config['mlp_config']:
                if not(isinstance(x,int)) or x<=0:
                    raise Exception('The number of hidden units per hidden layers for the mlp config layers has to be a positive integer.')
        except:
            raise ValueError('The definition of the feed forward layers must be an iterable with integers, e.g. [64, 64] for a two layer MLP with 64 units in each layer.')

        if not(isinstance(config['max_cutting_length'], int)) or config['max_cutting_length']<=0:
            raise Exception('The maximum length for the temporal sequence has to be a positive integer.')

        if not(isinstance(config['cl_every_epoch'], int)) or config['cl_every_epoch']<=0:
            raise Exception('The number of epochs has to be a positive integer.')

        if not(isinstance(config['deactivate_TF'], int)) or config['deactivate_TF']<0:
            raise Exception('The number of epochs after which deactivating the teacher forcing technique has to be a non negative integer.')

        if config['prob']>1 or config['prob']<=0:
            raise Exception('The probability of keeping the units in the variational layers should be in (0,1] interval.')

        if config['lr'] <=0:
            raise Exception('The learning rate should be >0.')

        if config['final_lr'] <0 or config['final_lr']>config['lr']:
            raise Exception('The final learning rate should be >0 and smaller than the initial learning rate.')

        if config['momentum'] <0 or config['momentum'] >= 1:
            raise Exception('The momentum must be in [0, 1)!')

        if config['scheduler'] not in(list_scheduler):
            raise Exception('This scheduler is not available\n Please, choose among {}.'.format(list_scheduler))

        if config['curriculum_learning'] not in(cl_states):
            raise Exception('The schedule %s for the curriculum learning is not available\n Please, choose among {}'.format(cl_states))

        if config['scheduler'] == 'step':
            if not(isinstance(config['num_lr_steps'], int)) or config['num_lr_steps']<=0 :
                raise Exception('The number of step size for the step decaying schedule of the learning rate has to be a positive integer.')

        if not config['act'] in activation_mapping.keys():
            raise Exception('The activation function %s is not available\n Please, choose among \'relu\', \'sigmoid\' or \'tanh\''%(config['act']))

        if save:
            path = os.path.join(output_path, dir_name)

            if not os.path.exists(path):
                os.makedirs(path)
            hyp_fname = os.path.join(path, 'hyper.json')

            with open(hyp_fname, 'w') as f:
                json.dump(hyper_par, f)
            f.close()

        config['act'] = activation_mapping[config['act']]

        return(config)

    def __obj_function(self, residuals):
        '''
        Internal use only:

        It computes the mean-squared error of the residuals.
        '''

        return ((residuals**2).sum(dim=0)).mean()

    def __forward(self, train_sequence, cutting_epoch, current_epoch, max_epoch, config_data=None):
        """
        Internal use only:

        A function that unrolles the lstm network, forwards the samples and returns the achieved value of the loss.
        """

        M = train_sequence.size()[1]

        if config_data is None:
            hidden_lstm, C_t = self.lstm.initHidden(M)
        else:
            hidden_lstm, C_t = self.lstm.initHidden_mlp(config_data, M)

        self.lstm.mask_generate(M)

        seq_pred = torch.tensor([], dtype=dtype, device=device)

        cutting_point = min(train_sequence.size()[0], cutting_epoch) #10

        if current_epoch <= max_epoch:

            for i in range(cutting_point-1):
                if not(torch.isnan(train_sequence[i]).all()):
                    tmp = train_sequence[i].clone()
                    tmp[torch.isnan(tmp)] = 0
                    pred, hidden_lstm, C_t  = self.lstm(tmp, hidden_lstm, C_t)#lstm(train_sequence[i], hidden_lstm, C_t)
                    pred.unsqueeze_(0)
                    seq_pred = torch.cat((seq_pred, pred), dim=0)
                else:
                    break
        else:
            pred = train_sequence[0].clone()
            pred[torch.isnan(pred)] = 0

            for i in range(cutting_point-1):
                if not(torch.isnan(train_sequence[i]).all()):
                    pred, hidden_lstm, C_t  = self.lstm(pred, hidden_lstm, C_t)#lstm(train_sequence[i], hidden_lstm, C_t)
                    pred[torch.isnan(pred)] = 0
                    pred.unsqueeze_(0)
                    seq_pred = torch.cat((seq_pred, pred), dim=0)
                    pred.squeeze_(0)
                else:
                    break

        seq_pred = torch.cat((seq_pred, torch.tensor(np.nan*np.ones(((cutting_point-1)-seq_pred.size()[0], seq_pred.size()[1] , 1)), dtype=dtype, device=device)), dim=0)
        residuals = seq_pred - train_sequence[1:cutting_point, :, :]
        residuals[torch.isnan(train_sequence[1:cutting_point, :, :])] = 0
        loss = self.__obj_function(residuals)

        return loss

    def __post_sample_TF(self, n_samples, samples, cutting_epoch, config_data=None):
        """
        Internal use only:

        A function that samples from the posterior distribution and estimates
        its variance, mean and log-likelihood (dropout-MC).
        """

        np.seterr(all='ignore')

        max_length = min(samples.size()[0], cutting_epoch)
        y_post_tensor = torch.zeros((n_samples, samples.size()[0]-1, samples.size()[1]))

        for ii in range(n_samples):
            M = samples.size()[1]
            #initialize the hidden states and cells of the multilayer LSTM
            if config_data is None:
                hidden_lstm, C_t = self.lstm.initHidden(M)
            else:
                hidden_lstm, C_t = self.lstm.initHidden_mlp(config_data,M)
            #refresh the list of masks
            self.lstm.mask_generate(M)

            seq_pred = torch.tensor([], dtype=dtype, device=device)

            for i in range(samples.size()[0]-1):#we can predict only till end
                if i<(max_length-1):
                    if not(torch.isnan(samples[i]).all()):
                        #check that also the next one is not zero--> this is not possible, so just overwrite it
                        pred, hidden_lstm, C_t  = self.lstm(samples[i], hidden_lstm, C_t)
                        pred.unsqueeze_(0)
                        seq_pred = torch.cat((seq_pred, pred), dim=0)
                    else:
                        break
                else:
                    if not(torch.isnan(samples[i]).all()):
                        #check that also the next one is not zero
                        pred, hidden_lstm, C_t  = self.lstm(pred.squeeze_(0), hidden_lstm, C_t)
                        pred.unsqueeze_(0)
                        seq_pred = torch.cat((seq_pred, pred), dim=0)
                    else:
                        break

            seq_pred = torch.cat((seq_pred, torch.tensor(np.nan*np.ones(((samples.size()[0]-1)-seq_pred.size()[0], seq_pred.size()[1] , 1)), dtype=dtype, device=device)), dim=0)
            y_post_tensor[ii, :, :] = seq_pred.squeeze(2)

        y_post_tensor = y_post_tensor.to('cpu')
        samples = samples.to('cpu')
        idx = torch.isnan(samples[1:, :, :]).squeeze(2)
        y_post_tensor[:,idx] = np.nan
        Y_post = y_post_tensor.data.numpy()
        mean = Y_post.mean(axis=0)
        #add a small offset to prevent nans
        std = Y_post.std(axis=0)
        std += 1e-10*np.ones(std.shape)
        log_likelihood = scipy.stats.norm.logpdf(samples[1:, :, :].squeeze(2).numpy(), mean, std)

        return mean, std, log_likelihood

    def __save_model(self):
        """
        Internal use only:

        A function that saves the model's paramters.
        """

        #parameters of the LSTM
        model_par = {}

        #ft
        ft_par = {}

        for ii, par in enumerate(self.lstm.ft):
            ft_par['w'+str(ii)] = par.weight.data.tolist()
            ft_par['b'+str(ii)] = par.bias.data.tolist()
        model_par['ft'] = ft_par

        #it
        it_par = {}

        for ii, par in enumerate(self.lstm.it):
            it_par['w'+str(ii)] = par.weight.data.tolist()
            it_par['b'+str(ii)] = par.bias.data.tolist()
        model_par['it'] = it_par

        #Ctild_t
        Ctild_par = {}

        for ii, par in enumerate(self.lstm.Ctild_t):
            Ctild_par['w'+str(ii)] = par.weight.data.tolist()
            Ctild_par['b'+str(ii)] = par.bias.data.tolist()
        model_par['Ctild_t'] = Ctild_par

        #ot
        ot_par = {}

        for ii, par in enumerate(self.lstm.ot):
            ot_par['w'+str(ii)] = par.weight.data.tolist()
            ot_par['b'+str(ii)] = par.bias.data.tolist()

        model_par['ot'] = ot_par
        #parameters for the mlp part

        mlp_par = {}

        ii = 0

        for par in self.lstm.mlp:
            try:
                mlp_par['w'+str(ii)] = par.weight.data.tolist()
                mlp_par['b'+str(ii)] = par.bias.data.tolist()
                ii += 1
            except:
                pass

        model_par['mlp'] = mlp_par

        #paramters for mlp_config
        mlpconfig_par = {}
        mlpN_par = {}

        for N, mlp_N in enumerate(self.lstm.mlp_config):

            ii = 0
            for par in mlp_N:
                try:
                    mlpN_par['w'+str(ii)] = par.weight.data.tolist()
                    mlpN_par['b'+str(ii)] = par.bias.data.tolist()
                    ii += 1
                except:
                    pass

            mlpconfig_par[str(N)] = mlpN_par
            mlpN_par = {}

        model_par['mlp_config'] = mlpconfig_par

        return model_par

    def load_model(self, hyper_par, par):
        '''
        This method loads a pre-trained model.

        Inputs:

            hyper_par: dictionary of hyperparameters to build the model.

            par: dictionary with model's parameters.
        '''

        activation_mapping = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }

        hyper_par['act'] = activation_mapping[hyper_par['act']]

        self.lstm = LSTM(1, hyper_par['hs'], 1, hyper_par['prob'], hyper_par['mlp'], self.config_size, hyper_par['mlp_config'], hyper_par['act'])
        #load the parameters'value
        self.lstm.init_par(par=par)

    def eval(self,n_samples=128, max_observed_epochs=5, bs=10):
        '''
        This method evaluates the posterior distribution of the model for training and test data
        and returns a dictionary with mean, std and log-likelihood for training and test data.

        Inputs:

            hyper_par: dictionary of hyperparameters to build the model.

            par: dictionary with model's parameters.

            n_samples: number of samples to draw from the posterior distribution.

            max_observed_epochs: number of epochs of the true learning curves that are fed into the model at test time (observed epochs).

            bs: batch size to be used for evaluation of the posterior.

        Outputs:

            dictionary with mean, standard deviation, log-likelihood and loss for test data.
        '''
        if self.lstm is None:
            raise Exception('A model needs to be trained or loaded from a pre-trained one!')

        res = {}

        N_test = self.test_data_t.shape[1]
        n_iters_test = int(np.ceil(N_test/bs))
        test_indices = np.arange(N_test)

        mean = np.asarray([])
        std = np.asarray([])
        log_likelihood = np.asarray([])

        print('SAMPLING FROM THE POSTERIOR\n')

        for i in range(n_iters_test):
            print('[{}/{}]'.format(i+1, n_iters_test))
            if self.config_size is not None:
                mean_batch, std_batch, log_likelihood_batch = self.__post_sample_TF(n_samples, self.test_data_t[:,test_indices[i*bs: min((i+1)*bs, N_test)]], int(max_observed_epochs), config_data=self.test_config_data_t[test_indices[i*bs: min((i+1)*bs, N_test)],:])
            else:
                mean_batch, std_batch, log_likelihood_batch = self.__post_sample_TF(n_samples, self.test_data_t[:,test_indices[i*bs: min((i+1)*bs, N_test)]], int(max_observed_epochs))
            if len(mean)>0:
                mean = np.concatenate((mean, mean_batch), axis=1)
                std = np.concatenate((std, std_batch), axis=1)
                log_likelihood = np.concatenate((log_likelihood, log_likelihood_batch), axis=1)
            else:
                mean = mean_batch
                std = std_batch
                log_likelihood = log_likelihood_batch

        residuals = mean - self.test_data_t[1:, :, 0].data.numpy()
        mse = (residuals**2).sum(axis=0).mean()

        res.update({'test mean': mean.tolist(), 'test std': std.tolist(), 'test log_likelihood':log_likelihood.tolist(), 'test mse': mse.tolist(), 'test data':self.test_data_t.data.tolist()})

        return res

    def train_vrnn(self, hyper_par=None, save=False, output_path='./saved_models', dir_name='model'):
        """
        A function that trains and tests a variational lstm.

        Inputs:

            hyper_par: a dictionary containing the hyperparameters.

            save: boolean. If true the model will be saved after training.

            output_path: path where a folder for saving the results and the model will be created.

            dir_name: name of directory where results and model will be saved.

        Outputs:

            A dictionary containg the training and test losses.
        """

        if hyper_par is None:
            hyper_par = {}

        hyper_par = self.__check_hyper(hyper_par, save, output_path, dir_name)

        bs = hyper_par['batch_size']

        N = self.train_data_t.shape[1]

        n_iters = int(np.ceil(N/bs))

        train_indices = np.arange(N)

        self.lstm = LSTM(1, hyper_par['hs'], 1, hyper_par['prob'], hyper_par['mlp'], self.config_size, hyper_par['mlp_config'], hyper_par['act'])

        #move the model on gpu if available
        if torch.cuda.is_available():
            self.lstm = self.lstm.cuda()

        optimizer = torch.optim.SGD(self.lstm.parameters(), lr=hyper_par['lr'], momentum=hyper_par['momentum'])

        scheduler = None

        res_dict = {}

        all_losses = []
        all_test_losses = []

        self.lstm.init_par()

        cutting_epoch = 0

        epoch = 0

        start_time = time.time()

        while(time.time()-start_time<hyper_par['budget']):

            epoch += 1

            if epoch>1:

                time_per_epoch = (time.time()-start_time)/(epoch-1)
                tot_epochs = np.ceil(hyper_par['budget']/time_per_epoch)

                if not(hyper_par['scheduler'] == 'const'):
                    for group in optimizer.param_groups:
                            group.setdefault('initial_lr', hyper_par['lr'])

                if hyper_par['scheduler'] == 'cos':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tot_epochs, eta_min=hyper_par['final_lr'], last_epoch=epoch-2)
                elif hyper_par['scheduler'] == 'step':
                    hyper_par['step_size'] = int(np.ceil(tot_epochs/(hyper_par['num_lr_steps']+1)))
                    hyper_par['gamma_step'] = np.power(hyper_par['final_lr']/hyper_par['lr'], 1./hyper_par['num_lr_steps'])
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hyper_par['step_size'], gamma=hyper_par['gamma_step'], last_epoch=epoch-2)
                elif hyper_par['scheduler'] == 'exp':
                    hyper_par['gamma_exp'] = np.power(hyper_par['final_lr']/hyper_par['lr'], 1./tot_epochs)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hyper_par['gamma_exp'], last_epoch=epoch-2)

            current_loss = 0

            random.shuffle(train_indices)

            if hyper_par['curriculum_learning']=='rand' and not((epoch-1)%hyper_par['cl_every_epoch']):
                cutting_epoch = random.randint(5, hyper_par['max_cutting_length'])
            elif hyper_par['curriculum_learning'] == 'linear':
                min_observed_epochs = min(hyper_par['max_cutting_length'], self.train_data_t.shape[0])
                cutting_epoch = min((int(( self.train_data_t.shape[0]- min_observed_epochs )/hyper_par['budget']*(time.time()-start_time) + min_observed_epochs),
                self.train_data_t.shape[0]))
            elif hyper_par['curriculum_learning']=='const':
                cutting_epoch = hyper_par['max_cutting_length']
            elif hyper_par['curriculum_learning']=='seq' and not((epoch-1)%hyper_par['cl_every_epoch']):
                cutting_epoch += 10
                cutting_epoch = min(cutting_epoch, hyper_par['max_cutting_length'])

            for i in range(n_iters):
                optimizer.zero_grad()

                if self.config_size is not None:
                    loss = self.__forward(self.train_data_t[:,train_indices[i*bs: min((i+1)*bs, N)]],
                                                cutting_epoch, epoch, max_epoch=hyper_par['deactivate_TF'], config_data=self.train_config_data_t[train_indices[i*bs: min((i+1)*bs, N)],:])
                else:
                    loss = self.__forward(self.train_data_t[:,train_indices[i*bs: min((i+1)*bs, N)]],
                                                cutting_epoch, epoch, max_epoch=hyper_par['deactivate_TF'])

                current_loss += loss.item()

                loss.backward()
                optimizer.step()

            if np.isnan(current_loss):
                break

            #test evaluation
            if self.config_size is not None:
                test_loss = self.__forward(self.test_data_t, cutting_epoch, epoch, max_epoch=hyper_par['deactivate_TF'], config_data=self.test_config_data_t)
            else:
                test_loss = self.__forward(self.test_data_t, cutting_epoch, epoch, max_epoch=hyper_par['deactivate_TF'])

            print('epoch {}, cutting point {}, training loss {}, test loss {}'.format(epoch, cutting_epoch, current_loss / (n_iters), test_loss.item()))

            all_losses.append(current_loss / (n_iters))
            all_test_losses.append(test_loss.item())

        res_dict['loss'] = all_losses
        res_dict['test loss'] = all_test_losses

        N_par = self.lstm.N_par_count()

        if save:
            par = self.__save_model()
            self.par = par

            path = os.path.join(output_path, dir_name)

            if not os.path.exists(path):
                os.makedirs(path)
            par_fname = os.path.join(path, 'par.json')

            with open(par_fname, 'w') as f:
                json.dump(par, f)

            f.close()

        return res_dict
