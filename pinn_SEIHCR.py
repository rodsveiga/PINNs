import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import time
from copy import deepcopy

class PINN_SEIHCR(nn.Module):
    '''
    Neural Network Class
    net_layer: list with the number of neurons for each network layer, [n_imput, ..., n_output]
    '''
    def __init__(self, 
                 s, 
                 e,
                 i,
                 h, 
                 c, 
                 r,
                 t, 
                 layers_size= [7, 20, 20, 20, 20], 
                 out_size = 6,
                 params_list= None):
        
        super(PINN_SEIHCR, self).__init__()

        #### Data
        self.s = s
        self.e = e
        self.i = i
        self.h = h
        self.c = c
        self.r = r
        self.t = t
        
        #### Initializing neural network 

        self.layers = nn.ModuleList()

        #### Initialize parameters (we are going to learn lambda)
        self.beta = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.gamma_L = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.gamma_IR = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.gamma_IH = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.f_IR = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.gamma_C = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.gamma_H = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.f_HC = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.f_CD = nn.Parameter(torch.zeros(1, requires_grad= True))

        for k in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[k], layers_size[k+1]))
            
        # Output layer
        self.out = nn.Linear(layers_size[-1], out_size)
        
        m_aux = 0
        
        for m in self.layers:
            
            if params_list is None:
                nn.init.normal_(m.weight, mean= 0, std= 1/np.sqrt(len(layers_size)))
                nn.init.constant_(m.bias, 0.0)
            else:
                m.weight = params_list[m_aux]
                m.bias = params_list[m_aux + 1]
                m_aux += 1
                
        if params_list is None:
            nn.init.normal_(self.out.weight, mean= 0, std= 1/np.sqrt(len(layers_size)))
            nn.init.constant_(self.out.bias, 0.0)
        else:
            self.out.weight = params_list[-2]
            self.out.bias = params_list[-1]

        #self.model = nn.ModuleDict()

        #self.optimizer = optim.SGD(params= self.model.parameters(), 
        #                           lr= 0.001, 
        #                           weight_decay= 0.0)


    #### Forward pass
       
    def forward(self, x):

        for layer in self.layers:

            # Activation function
            x = torch.tanh(layer(x))
        
        # Last layer: we could choose a different functionsoftmax
        #output= F.softmax(self.out(x), dim=1)
        output= self.out(x)
        
        return output

    #### Derivative

    def delta(self, x):
        list_ = []
        list_.append(0)
        for j in range(len(x) - 1):
            list_.append(x[j+1].item() - x[j].item())
        return torch.Tensor(list_)  


    #### Net NS

    def net(self, s, e, i, h, c, r, t):

        beta = self.beta
        gamma_L = self.gamma_L 
        gamma_IR = self.gamma_IR 
        gamma_IH = self.gamma_IH 
        f_IR = self.f_IR 
        gamma_C = self.gamma_C 
        gamma_H = self.gamma_H 
        f_HC = self.f_HC 
        f_CD =  self.f_CD
        
        X = torch.stack((s, e, i, h, c, r, t), 1)

        # Parametrize [psi, p] as neural network
        seirhcr_model = self.forward(X)
        #print(psi_p)

        ### Check this on a notebook
        s_model = seirhcr_model[:, 0:1]
        e_model = seirhcr_model[:, 1:2]
        i_model = seirhcr_model[:, 2:3]
        h_model = seirhcr_model[:, 3:4]
        c_model = seirhcr_model[:, 4:5]
        r_model = seirhcr_model[:, 5:6]
       
 

        dsdt = self.delta(s)
        dedt = self.delta(e)
        didt = self.delta(i)
        dhdt = self.delta(h)
        dcdt = self.delta(c)
        drdt = self.delta(r)


        ### PINN

        f_s = dsdt + beta*s_model*i_model
        f_e = dedt - beta*s_model*i_model + gamma_L*e_model
        f_i = didt - gamma_L*e_model + gamma_IR*f_IR*i_model + gamma_IH*(1 - f_IR)*i_model
        f_h = dhdt - gamma_IH*(1 - f_IR)*i_model - gamma_C*(1 - f_CD)*c_model + gamma_H*h_model
        f_c = dcdt - gamma_H*f_HC*h_model + gamma_C*c_model
        f_r = drdt - gamma_IR*f_IR*i_model - gamma_H*(1 - f_HC)*h_model

        return s_model, e_model, i_model, h_model, c_model, r_model, f_s, f_e, f_i, f_h, f_c , f_h


    def loss(self, s, e, i, h, c, r,
             s_hat, e_hat, i_hat, h_hat, c_hat, r_hat, 
             f_s, f_e, f_i, f_h, f_c, f_r):

        error1 = torch.mean(torch.square(s - s_hat)) + torch.mean(torch.square(e - e_hat))
        error2 = torch.mean(torch.square(i - i_hat)) + torch.mean(torch.square(h - h_hat))
        error3 = torch.mean(torch.square(c - c_hat)) + torch.mean(torch.square(r - r_hat))
        regul1 = torch.mean(torch.square(f_s)) + torch.mean(torch.square(f_e))
        regul2 = torch.mean(torch.square(f_i)) + torch.mean(torch.square(f_h))
        regul3 = torch.mean(torch.square(f_c)) + torch.mean(torch.square(f_r))

        return error1 + error2 + error3 + regul1 + regul2 + regul3


    def get_gradient(self, f, x):
        """ computes gradient of tensor f with respect to tensor x """
        assert x.requires_grad

        x_shape = x.shape
        f_shape = f.shape
        f = f.view(-1)

        x_grads = []
        for f_val in f:
            if x.grad is not None:
                x.grad.data.zero_()
            f_val.backward(retain_graph=True)
            if x.grad is not None:
                x_grads.append(deepcopy(x.grad.data))
            else:
                # in case f isn't a function of x
                x_grads.append(torch.zeros(x.shape).to(x))
        output_shape = list(f_shape) + list(x_shape)
        return torch.cat((x_grads)).view(output_shape)


    def train(self, epochs):

        #s, e, i, h, c, r, t

        t0 = time.time()
        
        for epoch in range(epochs):

            s_model, e_model, i_model, h_model, c_model, r_model, f_s, f_e, f_i, f_h, f_c, f_r = self.net(self.s,
                                                                                                          self.e, 
                                                                                                          self.i, 
                                                                                                          self.h, 
                                                                                                          self.c,
                                                                                                          self.r, 
                                                                                                          self.t)
            loss_ = self.loss(self.s, self.e, self.i, self.h, self.c, self.r, 
                             s_model, e_model, i_model, h_model, c_model, r_model, 
                             f_s, f_e, f_i, f_h, f_c, f_r)

            loss_print  = loss_

            self.optimizer.zero_grad()   # Clear gradients for the next mini-batches
            loss_.backward()         # Backpropagation, compute gradients
            self.optimizer.step()

            t1 = time.time()


            ### Training status
            print('Epoch %d, Loss= %.10f, Time= %.4f' % (epoch, 
                                                        loss_print,
                                                        t1-t0))