import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

n_actions = 2

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def predict_probs(states, model):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :param model: torch model
    :returns: numpy array of shape [batch, n_actions]
    """
    # convert states, compute logits, use softmax to get probability

    # YOUR CODE GOES HERE
    states = torch.from_numpy(states)
    with torch.no_grad(): 
        logits = model(states)
        probs = F.softmax(logits, dim=1)
    assert probs is not None, "probs is not defined"

    return probs.numpy()

def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    # YOUR CODE GOES HERE
    if len(rewards) == 0:
      return []
    else:
      t = len(rewards) - 1
      G = [0 for i in range (len(rewards))]
      G[t] = rewards[t]
      while t > 0:
        G[t-1] = rewards[t-1] + gamma*G[t]
        t -= 1
      cumulative_rewards = G
      assert cumulative_rewards is not None, "cumulative_rewards is not defined"

      return cumulative_rewards

def get_loss(logits, actions, rewards, n_actions=n_actions, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    """
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
    #print(f"actions {actions}\n")
    #print(f"logits {logits}\n")

    probs = F.softmax(logits, dim=1)
    #print(f"probs {probs}\n")
    assert probs is not None, "probs is not defined"

    log_probs = torch.log(probs)
    #print(f"log_probs {log_probs}\n")
    assert log_probs is not None, "log_probs is not defined"

    assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        "please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    log_probs_for_actions = [log_probs[i][actions[i]] for i in range (len(actions))]
    #print(f"log_probs_for_action {log_probs_for_actions}\n")
    assert log_probs_for_actions is not None, "log_probs_for_actions is not defined"
    J_hat = sum([log_probs_for_actions[i] * cumulative_returns[i] for i in range(len(actions))]) / len(actions)
    assert J_hat is not None, "J_hat is not defined"
    
    # Compute loss here. Don't forget entropy regularization with `entropy_coef`
    entropy = -(torch.sum(probs*log_probs, dim = 1)).mean()
    assert entropy is not None, "entropy is not defined"
    loss = -J_hat - entropy_coef*entropy
    assert loss is not None, "loss is not defined"

    return loss