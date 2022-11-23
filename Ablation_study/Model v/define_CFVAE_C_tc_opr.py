import numpy as np
import torch
from pylab import rcParams
import matplotlib.pyplot as plt
import os.path
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from model_CFVAE_C_tc_opr import VAE_model, reparam, KL, Discriminator
from sklearn.manifold import TSNE
import torch.nn as nn
import argparse
import os
from load_dataset import load_data


def run_CFVAE_C_tc_opr():
    parser = argparse.ArgumentParser(description='VAE for Counterfactual fairness')
    # arguments for optimization
    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--sensitive_latent_size', default=1, type=int)
    parser.add_argument('--target_latent_size', default=3, type=int)
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

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    for train_n in range(config.train_num):

        model = VAE_model(input_size=config.input_size, sensitive_latent_size=config.sensitive_latent_size, target_latent_size=config.target_latent_size, model_name=config.model_name)
        model.train()
        model.double()
        model = model.cpu()
        print('The structure of our model is shown below: \n')
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        model_d = Discriminator(target_latent_size=config.target_latent_size)
        model_d.train()
        model_d.double()
        model_d = model_d.cpu()
        print('The structure of our model_d is shown below: \n')
        print(model_d)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=config.lr)

        loss_epoch_z = []

        for epoch_number in range(config.train_epochs):

            print('epoch number:', epoch_number)

            for (data, label) in train_loader:
                label = label.double().cpu()
                data = data.double().cpu()

                if config.model_name == 'VAE': # Original version of VAE
                    q_zx_mean, q_zx_log_sigma = model.encoder_x(data[:,1:])
                    zx = reparam(q_zx_mean, q_zx_log_sigma).double()

                    D_z = model_d(zx)
                    vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                    zx_ = torch.zeros(zx.size()[0],3).double().cpu()

                    I = torch.eye(3).double().cpu()
                    A = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).double().cpu()
                    c = torch.inverse(I - A.t()).double().cpu()

                    for i in range(zx.size()[0]):
                        x = zx[i].reshape((3, 1)).double().cpu()
                        y = torch.mm(c, x).double().cpu()
                        zx_[i] = y.reshape((1, 3)).double().cpu()


                    reconst_zx = model.decoderX(zx_).double()

                    q_za_mean, q_za_log_sigma = model.encoder_a(data[:, 0:1])
                    za = reparam(q_za_mean, q_za_log_sigma)
                    reconst_za = model.decoderA(za)

                    reconst_loss_x = torch.nn.functional.mse_loss(reconst_zx,data[:,1:])
                    reconst_loss_a = torch.nn.functional.mse_loss(reconst_za,data[:,0:1])
                    kl_loss = KL(q_zx_mean, q_zx_log_sigma)
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    orth_loss = torch.mean(torch.abs(cos(data[:,0:1], zx_)))

                    loss = kl_loss + reconst_loss_x + reconst_loss_a + 10*vae_tc_loss + orth_loss

                model.zero_grad()
                model_d.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_d.step()

            print('kl_loss:', kl_loss)
            print('reconst_loss_x:', reconst_loss_x)
            print('reconst_loss_a:', reconst_loss_a)
            print('tc_loss:', vae_tc_loss)
            print('orth_loss:', orth_loss)
            print('batch loss:', loss)

            loss_epoch_z.append(loss.item())


        torch.save(model, './model/CFAVE_C_tc_opr')

if __name__ == '__main__':
        run_CFVAE_C_tc_opr()



