import numpy as np
import random

import torch
import torch.nn as nn

import torch.distributions.bernoulli as Bernoulli

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

class LSTM(nn.Module):
    '''
    This class creates an instantion of a multilayer LSTM
    followed by a multilayer MLP for regression.
    If the configuration vector is available, then this is information is fed thorugh
    additional MLPs to each of the stacked LSTMs.

    Everything is parametric, therefore it is possible to
    adjust the number of stacked LSTM networks, as well as
    their hidden size, and the number of layers, hidden units
    and non linearities for the MLP final part.

    Inputs:

        input_size: dimensionality of the input vector.

        hidden_size: number of units for the stacked lstms, i.e. [32, 64] two stacked LSTMs with 32 and 64 hidden units respectively.

        output_size: dimensionality of the output vector.

        model_prob: probability for the Bernoulli distribution
        (it corresponds to the probability of keeping a certain unit).

        mlp_layers: list containing the number of units per mlp-layer, i.e. [32, 64] corresponds to a 2-hidden layers MLP where each hidden layer has 32 and 64 units respectively.

        config_size: size of the configuration vector.

        mlp_config: list containing the number of units per mlp-layer for the mlp used to elaborate the configuration vector, i.e. [32, 64] corresponds to a 2-hidden layers MLP where each hidden layer has 32 and 64 units respectively.

        activation: activation function for the hidden-layers in the mlps.
    '''

    def __init__(self, input_size, hidden_size, output_size, model_prob, mlp_layers=[], config_size=None, mlp_config=[], activation=nn.Sequential()):
        '''
        create a variational LSTM model.
        '''

        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.z_mask = []
        self.ft = nn.ModuleList()
        self.it = nn.ModuleList()
        self.Ctild_t = nn.ModuleList()
        self.ot = nn.ModuleList()

        for ii in range(len(hidden_size)):
            if ii == 0:
                self.ft.append(nn.Linear(input_size+hidden_size[ii], hidden_size[ii]))
                self.it.append(nn.Linear(input_size+hidden_size[ii], hidden_size[ii]))
                self.Ctild_t.append(nn.Linear(input_size+hidden_size[ii], hidden_size[ii]))
                self.ot.append(nn.Linear(input_size+hidden_size[ii], hidden_size[ii]))
            else:
                self.ft.append(nn.Linear(hidden_size[ii-1]+hidden_size[ii], hidden_size[ii]))
                self.it.append(nn.Linear(hidden_size[ii-1]+hidden_size[ii], hidden_size[ii]))
                self.Ctild_t.append(nn.Linear(hidden_size[ii-1]+hidden_size[ii], hidden_size[ii]))
                self.ot.append(nn.Linear(hidden_size[ii-1]+hidden_size[ii], hidden_size[ii]))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        mlp_layers.insert(0, hidden_size[-1])
        mlp_layers.append(output_size)
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.model_bernoulli = Bernoulli.Bernoulli(probs=model_prob)
        #self.dropout = nn.Dropout(1-model_prob)
        #remove dropout in the mlp layer in order to make the training process easier
        self.dropout = nn.Dropout(0)
        mlp_list = []

        for ii in range(len(self.mlp_layers)-1):
            mlp_list.append(self.dropout)
            mlp_list.append(nn.Linear(self.mlp_layers[ii], self.mlp_layers[ii+1]))
            if ii+2 < len(self.mlp_layers):
                mlp_list.append(self.activation)
        self.mlp = nn.Sequential(*mlp_list)

        self.mlp_config = nn.Sequential()

        if config_size is not None:

            mlp_config.insert(0, config_size)
            count = 0
            for ii in range(len(hidden_size)):
                count += 1
                mlp_config_ii = []

                for jj in range(len(mlp_config)-1):
                    mlp_config_ii.append(nn.Linear(mlp_config[jj], mlp_config[jj+1]))
                    mlp_config_ii.append(self.activation)

                mlp_config_ii.append(nn.Linear(mlp_config[-1], hidden_size[ii]))

                self.mlp_config.add_module('mlp {}'.format(count), nn.Sequential(*mlp_config_ii))

    def forward(self, input, h_list, Ct_list):
        '''
        This function forwards the input through the model and computes the output, hidden state and current state of the model.
        '''

        for ii in range(len(self.hidden_size)):
            if(ii==0):
                combined = torch.cat((input, h_list[ii]*self.z_mask[ii]), 1)

            else:
                combined = torch.cat((h_list[ii-1]*self.z_mask[ii-1], h_list[ii]*self.z_mask[ii]), 1)

            C_t = self.sigmoid(self.ft[ii](combined))*Ct_list[ii] + self.sigmoid(self.it[ii](combined))*self.tanh(self.Ctild_t[ii](combined))
            output_h = self.sigmoid(self.ot[ii](combined))
            hidden_t = output_h*self.tanh(C_t)
            Ct_list[ii] = C_t
            h_list[ii] = hidden_t
        output = self.mlp(hidden_t)

        return output, h_list, Ct_list

    def init_par(self, par=None):
        '''
        This function initialized the parameters of the model. If a dictionary par is provided,
        then the parameters of the model will be initialized with those provided in the dictionary.
        Otherwise, the xavier initialization will be used.
        '''

        if par is None:
            for p in LSTM.parameters(self):
                if len(p.shape)==2:
                    torch.nn.init.xavier_uniform_(p).to(device)
                if len(p.shape)==1:
                    p.data.fill_(0.01).to(device)
        else:
            #keys of par: 'ft', 'it', 'Ctild_t', 'ot', 'mlp', 'mlp_config'
            for ii, par_ii in enumerate(self.ft):
                par_ii.weight.data = torch.FloatTensor(par['ft']['w'+str(ii)])
                par_ii.bias.data = torch.FloatTensor(par['ft']['b'+str(ii)])
            for ii, par_ii in enumerate(self.it):
                par_ii.weight.data = torch.FloatTensor(par['it']['w'+str(ii)])
                par_ii.bias.data = torch.FloatTensor(par['it']['b'+str(ii)])
            for ii, par_ii in enumerate(self.ot):
                par_ii.weight.data = torch.FloatTensor(par['ot']['w'+str(ii)])
                par_ii.bias.data = torch.FloatTensor(par['ot']['b'+str(ii)])
            for ii, par_ii in enumerate(self.Ctild_t):
                par_ii.weight.data = torch.FloatTensor(par['Ctild_t']['w'+str(ii)])
                par_ii.bias.data = torch.FloatTensor(par['Ctild_t']['b'+str(ii)])
            count=0
            for par_ii in self.mlp:
                try:
                    par_ii.weight.data = torch.FloatTensor(par['mlp']['w'+str(count)])
                    par_ii.bias.data = torch.FloatTensor(par['mlp']['b'+str(count)])
                    count += 1
                except:
                    pass

            for n, config_ii in enumerate(self.mlp_config):
                ii = 0
                for par_ii in config_ii:

                    try:
                        par_ii.weight.data = torch.FloatTensor(par['mlp_config'][str(n)]['w'+str(ii)])
                        par_ii.bias.data = torch.FloatTensor(par['mlp_config'][str(n)]['b'+str(ii)])
                        ii += 1
                    except:
                        pass

    def N_par_count(self):
        '''
        This function counts the number of parameters in the model.
        '''

        N_par = 0
        for p in LSTM.parameters(self):
            if len(p.shape)==2:
                N_par += p.shape[0]*p.shape[1]
            else:
                N_par += p.shape[0]

        return N_par

    def initHidden(self, mini_batches):
        '''
        Initialize the hidden states and the states with zeros.
        '''

        h_init = []
        Ct_init = []

        for ii in range(len(self.hidden_size)):
            h_init.append(torch.zeros(mini_batches, self.hidden_size[ii], dtype=dtype, device=device))
            Ct_init.append(torch.zeros(mini_batches, self.hidden_size[ii], dtype=dtype, device=device))

        return h_init, Ct_init

    def initHidden_mlp(self, config_input, mini_batches):
        '''
        The hidden states are not initialized with zeros but by feeding the
        configuration vectors through a dedicated mlp (mlp_config).
        '''

        h_init = []
        Ct_init = []

        for ii in range(len(self.hidden_size)):

            h_init.append(self.mlp_config[ii](config_input))
            Ct_init.append(torch.zeros(mini_batches, self.hidden_size[ii], dtype=dtype, device=device))

        return h_init, Ct_init

    def mask_generate(self, mini_batches):
        '''
        This function draws masks from a Bernoulli distribution for the variational layers in the LSTMs.
        '''

        self.z_mask = []

        for ii in range(len(self.hidden_size)):
            self.z_mask.append(self.model_bernoulli.sample(sample_shape=torch.Size([mini_batches,self.hidden_size[ii]])).to(device=device))
