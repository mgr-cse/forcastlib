
import pyro
import pyro.distributions as dist

from torch import nn
from pyro.nn import PyroModule

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

from pyro.nn import PyroSample

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive

from tqdm.auto import trange


class BayesianRegression(PyroModule):
    # init model to spec
    def __init__(self, in_features, out_features, w_prior=None, bias_prior=None, noise_prior=None):
        super().__init__()
        # required to fit
        self.in_features = in_features
        self.out_features = out_features
        #pyro model spec
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        # change weight priors here
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

        # set priors
        if w_prior != None:
            self.linear.weight = PyroSample(w_prior.expand([out_features, in_features]).to_event(2))
        if bias_prior != None:
            self.linear.bias = PyroSample(bias_prior.expand([out_features]).to_event(1))
        if noise_prior != None:
            self.noise_prior = noise_prior
        else: self.noise_prior = None

        
    # required override
    def forward(self, x, y=None):
        # noise prior
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        if self.noise_prior != None:
            sigma = pyro.sample("sigma", self.noise_prior)
        mean = self.linear(x).squeeze(-1)
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