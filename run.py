import argparse
from dqn_agent import SnakeDQNAgent
import torch

def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=777, help="random seed for reproducibility"
    )
    # parser.add_argument(
    #     "--integration-test",
    #     dest="integration_test",
    #     action="store_true",
    #     help="for integration test",
    # )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./configs/dqn.yaml",
        help="config path",
    )
    # parser.add_argument(
    #     "--test", dest="test", action="store_true", help="test mode (no training)"
    # )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="load the saved model and optimizer at the beginning",
    )
    # parser.add_argument(
    #     "--off-render", dest="render", action="store_false", help="turn off rendering"
    # )
    # parser.add_argument(
    #     "--render-after",
    #     type=int,
    #     default=0,
    #     help="start rendering after the input number of episode",
    # )
    parser.add_argument(
        "--log", dest="log", action="store_true", default=True, help="turn on logging"
    )
    parser.add_argument(
        "--save-period", type=int, default=100, help="save model period"
    )
    parser.add_argument(
        "--episode-num", type=int, default=10000, help="total episode num"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=1000, help="max episode step"
    )
    # parser.add_argument(
    #     "--interim-test-num",
    #     type=int,
    #     default=10,
    #     help="number of test during training",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()
    agent = SnakeDQNAgent(args)
    agent.initialise()

    # agent.train()
    agent.train_multi_proc(20)
