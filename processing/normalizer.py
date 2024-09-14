"""Set of wrappers for normalizing actions and observations."""
import numpy as np
import torch


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    def __init__(self, num_tasks, epsilon=1e-8):
        self.mean = torch.zeros(num_tasks)
        self.var = torch.ones(num_tasks)
        self.count = torch.zeros(num_tasks)
        self.epsilon = epsilon

    def update(self, x, task_id):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count, task_id)

    def update_from_moments(self, batch_mean, batch_var, batch_count, task_id):
        """Updates from batch mean, variance and count moments."""
        self.mean[task_id], self.var[task_id], self.count[task_id] = update_mean_var_count_from_moments(
            self.mean[task_id], self.var[task_id], self.count[task_id], batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation():
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, num_tasks, epsilon = 1e-8):
        self.obs_rms = RunningMeanStd(num_tasks)
        self.epsilon = epsilon

    def normalize(self, obs, task_id):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs, task_id)
        return (obs - self.obs_rms.mean[task_id]) / np.sqrt(self.obs_rms.var[task_id] + self.epsilon)


class NormalizeReward():
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(self, num_tasks, gamma = 0.99, epsilon = 1e-8):

        self.return_rms = RunningMeanStd(num_tasks)
        self.returns = np.zeros(num_tasks)
        self.gamma = gamma
        self.epsilon = epsilon

    def normalize(self, rews, term, task_id):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.returns = self.returns * self.gamma * (1 - term) + rews
        self.return_rms.update(self.returns, task_id)
        return rews / np.sqrt(self.return_rms.var[task_id] + self.epsilon)