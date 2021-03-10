import json
import argparse
from src.maze import Maze


def instanciate_from_string(module_path):
    path_list = module_path.split(".")
    class_name = path_list[-1]
    directory_of_module = ".".join(path_list[:-1])
    module = __import__(directory_of_module, fromlist=[class_name])
    return getattr(module, class_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train agent to navigate a maze specified in the instruction file.')
    parser.add_argument('--instruction', metavar='I', type=str,
                        help='path to instruction file', default="../instructions/prim_maze.json")

    args = parser.parse_args()
    # Load instruction file:
    with open(args.instruction, "r") as file:
        instruction = json.load(file)

    obstacles = instanciate_from_string(instruction["obstacles"]["class"])(**instruction["obstacles"]["kwargs"])
    maze = Maze(obstacles=obstacles)
    learner = instanciate_from_string(instruction["learner"]["class"])(maze, **instruction["learner"]["kwargs"])
    learner.learn(**instruction["learner__learn_kwargs"])
