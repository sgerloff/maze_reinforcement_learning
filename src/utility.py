import os


def get_project_dir():
    file_path = os.path.abspath(__file__)  # <project_dir>/src/utility.py
    file_dir = os.path.dirname(file_path)  # <project_dir>/src
    return os.path.dirname(file_dir)  # <project_dir>
