# configuration.py
from utils import AttrDict


config = AttrDict(
    gamma=0.99,
    lam=0.95,
    eps_clip=100,
    k_epoch=4,
    lr=1e-4,
    c1=1,
    c2=0.5,
    c3=1e-3,
    num_env=8,
    seq_length=16,
    batch_size=64,
    minibatch_size=16,
    hidden_size=128,
    train_env_steps=1000000,
    num_eval_episode=100,
)