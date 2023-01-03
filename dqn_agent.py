import time
import random
from datetime import datetime

import yaml
import numpy as np
import torch

from rl_algorithms import DQNAgent
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper

from dqn_learner import CustomDQNLearner
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from config import create_object, EnvInfo, LogCfg

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class Agent(DQNAgent):
    def __init__(self):
        env_name = "snake_ai"

        # test settings
        self.is_log = False
        # self.load_from = "checkpoint/snake_ai/DQNAgent/2022-12-27_21:58:52/ep_40.pt"
        self.load_from = None
        self.is_test = False
        self.save_period = 500
        self.episode_num = 10000
        self.max_episode_steps = 10000
        self.interim_test_num = None  # 100

        # hyper parameters
        cfg_path = "config/dqn.yaml"
        cfg = self.get_cfg(cfg_path)
        self.hyper_params = cfg.hyper_params
        self.optim_cfg = cfg.learner_cfg.optim_cfg
        self.head_cfg = cfg.learner_cfg.head
        self.loss_type = cfg.learner_cfg.loss_type
        self.env_info = EnvInfo(env_name)
        self.log_cfg = LogCfg(
            cfg.type, datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), env_name, cfg_path
        )

        self.per_beta = self.hyper_params.per_beta
        self.use_n_step = self.hyper_params.n_step > 1

        if self.head_cfg.configs.use_noisy_net:
            self.max_epsilon = 0.0
            self.min_epsilon = 0.0
            self.epsilon = 0.0
        else:
            self.max_epsilon = self.hyper_params.max_epsilon
            self.min_epsilon = self.hyper_params.min_epsilon
            self.epsilon = self.hyper_params.max_epsilon

        # initialise a learner
        self.learner = None
        self._initialize()
        self.game = SnakeGameAI()

        #
        self.curr_state = np.zeros(1)
        self.episode_step = 0
        self.i_episode = 0
        self.total_step = 0

    @staticmethod
    def get_cfg(cfg_path):
        # Open the YAML file and parse it
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f)
        cfg = create_object(data)
        return cfg

    def _initialize(self):
        """Initialize non-common things."""
        if not self.is_test:
            # replay memory for a single step
            self.tmp_memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
            )
            self.memory: PrioritizedBufferWrapper = PrioritizedBufferWrapper(
                self.tmp_memory, alpha=self.hyper_params.per_alpha
            )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

        self.learner = CustomDQNLearner(
            self.hyper_params,
            self.log_cfg,
            self.log_cfg.env_name,
            self.optim_cfg,
            self.head_cfg,
            self.loss_type,
            False,
            None,
        )

    def get_state(self, game):
        """
        state (12 Values)
        [ length, danger straight, danger right, danger left,

        direction left, direction right,
        direction up, direction down

        food left,food right,
        food up, food down]
        """
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Current length
            len(game.snake),
            # Danger Straight
            (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d))
            or (dir_l and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_r)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger Left
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food Location
            game.food.x < game.head.x,  # food is in left
            game.food.x > game.head.x,  # food is in right
            game.food.y < game.head.y,  # food is up
            game.food.y > game.head.y,  # food is down
        ]
        return np.array(state, dtype=int)

    def select_action(self, state):
        self.curr_state = state
        final_move = [0, 0, 0]
        r = np.random.random()
        if not self.is_test and self.epsilon > r:
            move = random.randint(0, 2)
        else:
            with torch.no_grad():
                state0 = torch.tensor(state, dtype=torch.float).cpu()
                # prediction = self.model(state0).cpu()  # prediction by model
                prediction = self.learner.dqn(state0).cpu()
                move = torch.argmax(prediction).item()

        return move

    def step(self, action):
        reward, done, score = self.game.play_step(action)
        next_state = self.get_state(self.game)
        info = score
        if not self.is_test:
            # if the last state is not a terminal state, store done as false
            done_bool = False if self.episode_step == self.max_episode_steps else done

            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def train(self):

        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")

        for self.i_episode in range(1, self.episode_num + 1):
            state = self.get_state(self.game)
            self.episode_step = 0
            losses = list()
            done = False

            score = 0

            t_begin = time.time()

            while not done:
                action = self.select_action(state)
                action = np.array(action)
                next_state, reward, done, score = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            info = self.learner.update_model(experience)
                            loss = info[0:2]
                            indices, new_priorities = info[2:4]
                            losses.append(loss)  # for logging
                            self.memory.update_priorities(indices, new_priorities)

                    # decrease epsilon
                    self.epsilon = max(
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon,
                    )

                    # increase priority beta
                    fraction = min(float(self.i_episode) / self.episode_num, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

                state = next_state[:]
                # score += reward

            self.game.reset()
            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)

            if self.i_episode % self.save_period == 0:
                self.learner.save_params(self.i_episode)
                # self.interim_test()

    def train_mp(self, q, i):

        if not self.is_test:
            # replay memory for a single step
            self.tmp_memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
            )
            self.memory: PrioritizedBufferWrapper = PrioritizedBufferWrapper(
                self.tmp_memory, alpha=self.hyper_params.per_alpha
            )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

        avg_time_costs = []
        scores = []
        state = self.get_state(self.game)
        self.episode_step = 0
        done = False
        t_begin = time.time()
        score = 0
        while not done and self.max_episode_steps > self.episode_step:
            action = self.select_action(state)
            action = np.array(action)
            next_state, reward, done, score = self.step(action)
            self.total_step += 1
            self.episode_step += 1
            state = next_state[:]
            scores.append(score)

        print(f"{i}, {self.episode_step=}")

        t_end = time.time()
        avg_time_costs.append((t_end - t_begin) / self.episode_step)

        # add all experiences to queues
        idx = len(self.memory)
        obs = self.memory.buffer.obs_buf[:idx]
        action = self.memory.buffer.acts_buf[:idx]
        reward = self.memory.buffer.rews_buf[:idx]
        next_obs = self.memory.buffer.next_obs_buf[:idx]
        done = self.memory.buffer.done_buf[:idx]
        q.put((obs, action, reward, next_obs, done, scores))


    def train_multi_proc(self, num_proc=4):

        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")


        for self.i_episode in range(1, self.episode_num + 1):
            print(f"*************** {self.i_episode=}")
            processes = []
            queues = []
            for i in range(num_proc):
                self_copy = Agent()
                # self_copy = deepcopy(self)
                q = mp.Queue()
                p = mp.Process(target=self_copy.train_mp, args=(q, i))
                processes.append(p)
                queues.append(q)
                p.start()

            # Wait for all processes to finish
            for p in processes:
                p.join()
            print('All processes have ended')

            # Get the results from the queue and add them to replay buffer
            score_proc = [None] * num_proc
            for i, q in enumerate(queues):
                transitions = q.get()
                for s, a, r, n_s, done, score in zip(*transitions):
                    a = np.array(a)
                    transition = (s, a, r, n_s, done)
                    self._add_transition_to_memory(transition)
                    score_proc[i] = score  # keep trac the last one
            score = np.max(score_proc)

            losses = list()
            if len(self.memory) >= self.hyper_params.update_starts_from:
                for _ in range(self.hyper_params.multiple_update):
                    experience = self.sample_experience()
                    info = self.learner.update_model(experience)
                    loss = info[0:2]
                    indices, new_priorities = info[2:4]
                    losses.append(loss)  # for logging
                    self.memory.update_priorities(indices, new_priorities)

                # decrease epsilon
                self.epsilon = max(
                    self.epsilon
                    - (self.max_epsilon - self.min_epsilon)
                    * self.hyper_params.epsilon_decay,
                    self.min_epsilon,
                )
                print(f"{self.epsilon=}")

                # increase priority beta
                fraction = min(float(self.i_episode) / self.episode_num, 1.0)
                self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)
                print(f"{fraction=}")

                for q in queues:
                    q.close()

            self.game.reset()

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, 0)
                # print(log_value)
                self.write_log(log_value)


if __name__ == "__main__":
    agent = Agent()
    agent.train()
