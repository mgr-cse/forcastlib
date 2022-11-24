import numpy as np
import torch

import pyro
from pyro.contrib.timeseries import IndependentMaternGP


class GaussianProcess:
    def __init__(self, kernel='Matern', nu=1.5) -> None:
        self.nu = nu

    def fit(self, data, T_train, init_lr=0.01, final_lr=0.0003, num_steps=300, beta1=0.5):
        _, obs_dim = data.shape
        self.gp = IndependentMaternGP(
        nu=1.5, obs_dim=obs_dim, length_scale_init=1.5 * torch.ones(obs_dim)).double()
        self.T_train = T_train

        adam = torch.optim.Adam(
            self.gp.parameters(),
            lr = init_lr,
            betas=(beta1, 0.999),
            amsgrad=True
        )

        gamma = (final_lr/init_lr) ** (1/num_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)


        # train
        for step in range(num_steps):
            loss = -self.gp.log_prob(data[0:T_train, :]).sum() / T_train
            loss.backward()
            adam.step()
            scheduler.step()

            if step % 10 == 0:
                print("step %03d  loss: %.3f" % (step, loss.item()))

            self.data = data

    def predict(self):
        T_onestep = len(self.data) - self.T_train
        T_train = self.T_train
        _, obs_dim = self.data.shape
        onestep_means, onestep_stds = np.zeros((T_onestep, obs_dim)), np.zeros(
            (T_onestep, obs_dim)
        )

        for t in range(T_onestep):
            dts = torch.tensor([1.0]).double()
            pred_dist = self.gp.forecast(self.data[0 : T_train + t, :], dts)
    
            onestep_means[t, :] = pred_dist.loc.data.numpy()
            onestep_stds[t, :] = pred_dist.scale.data.numpy()
        
        return (onestep_means, onestep_stds)


