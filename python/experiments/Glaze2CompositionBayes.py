
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import GlazeCompositionDataset
import pyro.distributions as dist

def model (data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the beta prior
    f = pyro.sample("latent_cao", dist.Beta(alpha0, beta0))
    for i in range(len(data)):
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

def guide (data):
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
        constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
        constraint=constraints.positive)
    pyro.sample("latent_cao", dist.Beta(alpha_q, beta_q))

# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 5000
# do gradient steps
for step in range(n_steps):
    svi.step(data)