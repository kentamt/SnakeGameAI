import os
from typing import Tuple, Union
from copy import deepcopy
from collections import OrderedDict
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
import rl_algorithms.common.helper_functions as common_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_sizes[0]).to(device)
        self.hidden_list = []
        # self.hidden = nn.Linear(hidden_size, hidden_size).to(device)
        for i, hidden_size in enumerate(hidden_sizes[1:]):
            self.hidden_list.append(nn.Linear(hidden_size, hidden_size).to(device))
        self.head = nn.Linear(hidden_sizes[-1], output_size).to(device)

    def forward(self, x):
        x = F.relu(self.linear(x))
        # x = F.relu(self.hidden(x))
        for i in range(len(self.hidden_list)):
            x = F.relu(self.hidden_list[i](x))
        x = self.head(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "log"
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class CustomDQNLearner(Learner):
    def __init__(
        self,
        hyper_params,
        log_cfg,
        env_name,
        optim_cfg,
        head_cfg,
        loss_type,
        is_test,
        load_from,
    ):

        Learner.__init__(self, hyper_params, log_cfg, env_name, is_test)

        self.hyper_params = hyper_params
        self.optim_cfg = optim_cfg
        self.head_cfg = head_cfg
        self.loss_type = loss_type
        self.load_from = load_from
        self.use_n_step = self.hyper_params.n_step > 1

        self._init_network()

    @staticmethod
    def _get_loss_fn(loss_name):
        module_name = "rl_algorithms"
        class_name = loss_name
        module = importlib.import_module(module_name)
        loss_class = getattr(module, class_name)
        return loss_class()

    def _init_network(self):
        """Initialize networks and optimizers."""
        # Networks
        self.dqn = DQN(
            self.head_cfg.configs.input_size,
            self.head_cfg.configs.hidden_sizes,
            self.head_cfg.configs.output_size,
        )
        self.dqn_target = DQN(
            self.head_cfg.configs.input_size,
            self.head_cfg.configs.hidden_sizes,
            self.head_cfg.configs.output_size,
        )
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        # loss
        self.loss_fn = self._get_loss_fn(self.loss_type.type)

        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # load the optimizer and model parameters
        if self.load_from is not None:
            self.load_params(self.load_from)

    def update_model(
        self, experience: Union[TensorTuple, Tuple[TensorTuple]]
    ) -> Tuple[torch.Tensor, torch.Tensor, list, np.ndarray]:  # type: ignore
        """Update dqn and dqn target."""

        if self.use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        weights, indices = experience_1[-3:-1]

        gamma = self.hyper_params.gamma

        dq_loss_element_wise, q_values = self.loss_fn(
            self.dqn, self.dqn_target, experience_1, gamma, self.head_cfg
        )

        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            gamma = self.hyper_params.gamma**self.hyper_params.n_step

            dq_loss_n_element_wise, q_values_n = self.loss_fn(
                self.dqn, self.dqn_target, experience_n, gamma, self.head_cfg
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps

        if self.head_cfg.configs.use_noisy_net:
            self.dqn.head.reset_noise()
            self.dqn_target.head.reset_noise()

        return (
            loss.item(),
            q_values.mean().item(),
            indices,
            new_priorities,
        )

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }
        Learner._save_params(self, params, n_episode)

    # pylint: disable=attribute-defined-outside-init
    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Learner.load_params(self, path)

        params = torch.load(path)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def get_state_dict(self) -> OrderedDict:
        """Return state dicts, mainly for distributed worker."""
        dqn = deepcopy(self.dqn)
        return dqn.cpu().state_dict()

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection, used only in grad cam."""
        return self.dqn
