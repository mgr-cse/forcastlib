
import pyro
import pyro.distributions as dist

import torch
from torch import nn
from pyro.nn import PyroModule
from copy import deepcopy

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

from pyro.nn import PyroSample

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive

from tqdm.auto import trange


class BayesianNN(PyroModule):
    # init model to spec
    def __init__(self, in_features, out_features, num_hiddens=[], w_prior=None, bias_prior=None, noise_prior=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hiddens = num_hiddens
        self.hidden_priors = []

        # handle hidden layers
        num_last = in_features
        if len(num_hiddens) > 0:
            hidden = PyroModule[nn.Linear](in_features, num_hiddens[0])
            hidden.weight = PyroSample(dist.Normal(0., .3).expand([num_hiddens[0], in_features]).to_event(2))
            if w_prior != None:
                hidden.weight = PyroSample(w_prior[0].expand([num_hiddens[0], in_features]).to_event(2))
            hidden.bias = PyroSample(dist.Normal(0., 10.).expand([num_hiddens[0]]).to_event(1))
            if bias_prior != None:
                hidden.bias = PyroSample(bias_prior[0].expand([num_hiddens[0]]).to_event(1))
            self.hidden_priors.append(hidden)

            for i in range(1, len(num_hiddens)):
                hidden = PyroModule[nn.Linear](num_hiddens[i-1], num_hiddens[i])
                hidden.weight = PyroSample(dist.Normal(0., .3).expand([num_hiddens[i], num_hiddens[i-1]]).to_event(2))
                if w_prior != None:
                    hidden.weight = PyroSample(w_prior[i].expand([num_hiddens[0], in_features]).to_event(2))
                hidden.bias = PyroSample(dist.Normal(0., 10.).expand([num_hiddens[i]]).to_event(1))
                if bias_prior != None:
                    hidden.bias = PyroSample(bias_prior[i].expand([num_hiddens[0]]).to_event(1))
                self.hidden_priors.append(hidden)
            num_last = num_hiddens[-1]
            

        # output layer
        self.linear = PyroModule[nn.Linear](num_last, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., .3).expand([out_features, num_last]).to_event(2))
        if w_prior != None:
            self.linear.weight = PyroSample(w_prior[len(num_hiddens)].expand([num_hiddens[0], in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))
        if bias_prior != None:
            hidden.bias = PyroSample(bias_prior[len(num_hiddens)].expand([num_hiddens[0]]).to_event(1))
        
    # required override
    def forward(self, x, y=None):
        # noise prior
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        
        # forward to hidden layer
        #print('fwd1')
        h = deepcopy(x)
        for hl in self.hidden_priors:
            h = hl(h).squeeze(-1)
        #output layer
        mean = self.linear(h).squeeze(-1)
        
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

    # train on data
    def fit(self, X_train, Y_train, elbo_itr=3000):
        self.guide = AutoDiagonalNormal(self)
        adam = pyro.optim.Adam({"lr": 0.03})
        svi = SVI(self, self.guide, adam, loss=Trace_ELBO())
        
        pyro.clear_param_store()
        bar = trange(elbo_itr)

        for epoch in bar:
            loss = svi.step(X_train, Y_train)
            bar.set_postfix(loss=f'{loss / X_train.shape[0]:.3f}')
            if (epoch % 100) == 0:
                print('ELBO loss', loss)
    
    # predict on data
    def predict(self, X_test, num_samples=500):
        predictive = Predictive(self, guide=self.guide, num_samples=num_samples)
        preds = predictive(X_test)
        mean = preds['obs'].T.detach().numpy().mean(axis=1)
        std = preds['obs'].T.detach().numpy().std(axis=1)
        # return man and standard deviation of predicted distributions
        return (mean, std)