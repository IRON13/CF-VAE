import numpy as np
import torch
from pylab import rcParams
import matplotlib.pyplot as plt
import os.path
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from model_CFVAE import VAE_model, reparam, KL
from sklearn.manifold import TSNE
import torch.nn as nn
import argparse
import os
from load_dataset import load_data


def run_CFVAE():
    parser = argparse.ArgumentParser(description='VAE for Counterfactual fairness')
    # arguments for optimization
    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--sensitive_latent_size', default=1, type=int)
    parser.add_argument('--target_latent_size', default=10, type=int)
    parser.add_argument('--input_size', default=6, type=int)
    parser.add_argument('--test_epochs', default=50, type=int)

    parser.add_argument('--model_name', default='VAE', type=str)

    parser.add_argument('--train_num', default=1, type=int)
    parser.add_argument('--dataset', default='data', type=str)

    config = parser.parse_args()

    if config.dataset == 'data':
        train_data, train_label, test_data, test_data_cf, test_label = load_data()

    train_dataset = data_utils.TensorDataset(train_data, train_label)
    test_dataset = data_utils.TensorDataset(test_data, test_label)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True) #30162
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True) #15060

    for train_n in range(config.train_num):

        model = VAE_model(input_size=config.input_size, sensitive_latent_size=config.sensitive_latent_size, target_latent_size=config.target_latent_size, model_name=config.model_name)
        model.train()
        model.double()
        model = model.cpu()
        print('The structure of our model is shown below: \n')
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        loss_epoch_z = []

        for epoch_number in range(config.train_epochs):

            print('epoch number:', epoch_number)

            for (data, label) in train_loader:
                label = label.double().cpu()
                data = data.double().cpu()

                if config.model_name == 'VAE': # Original version of VAE
                    q_zx_mean, q_zx_log_sigma = model.encoder_x(data[:,1:])
                    zx = reparam(q_zx_mean, q_zx_log_sigma).double()

                    # zx_ = torch.zeros(zx.size()[0],2).double().cpu()
                    #
                    # I = torch.eye(2).double().cpu()
                    # A = torch.tensor([[0, 0], [0, 0]]).double().cpu()
                    # c = torch.inverse(I - A.t()).double().cpu()
                    #
                    # for i in range(zx.size()[0]):
                    #     x = zx[i].reshape((2, 1)).double().cpu()
                    #     y = torch.mm(c, x).double().cpu()
                    #     zx_[i] = y.reshape((1, 2)).double().cpu()


                    reconst_zx = model.decoderX(zx).double()

                    q_za_mean, q_za_log_sigma = model.encoder_a(data[:, 0:1])
                    za = reparam(q_za_mean, q_za_log_sigma)
                    reconst_za = model.decoderA(za)

                    reconst_loss_x = torch.nn.functional.mse_loss(reconst_zx,data[:,1:])
                    reconst_loss_a = torch.nn.functional.mse_loss(reconst_za,data[:,0:1])
                    kl_loss = KL(q_zx_mean, q_zx_log_sigma)

                    loss = kl_loss + reconst_loss_x + reconst_loss_a

                model.zero_grad()
                loss.backward()
                optimizer.step()

            print('kl_loss:', kl_loss)
            print('reconst_loss_x:', reconst_loss_x)
            print('reconst_loss_a:', reconst_loss_a)
            print('batch loss:', loss)

            loss_epoch_z.append(loss.item())


        torch.save(model, './model/CFAVE')


if __name__ == '__main__':
        run_CFVAE()



