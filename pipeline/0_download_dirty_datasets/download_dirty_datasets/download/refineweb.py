
import logging
import os
import subprocess
import preprocessing
import pathlib
from datasets import load_dataset


ROOT_PATH = pathlib.Path(preprocessing.__path__[0]).resolve().parent
SCRIPT_PATH = os.path.join(ROOT_PATH, "scripts")

def __execute_download(download_file: str, output_file_path: str, dataset_root: str) -> None:
    print(f"Downloading {download_file} to {output_file_path}")

    current_dir = os.getcwd()

    # Change directory in order to use git lfs
    os.chdir(dataset_root)
    subprocess.run(["git", "lfs", "pull", "--include", f"{download_file}"], check=True)
    os.chdir(current_dir)

    with open(f"{dataset_root}/data/{download_file}", 'rb') as input_file:
        with open(output_file_path, 'ab') as output_file:
            output_file.write(input_file.read())


def download_dataset(output_base: str = "output", index_from: int = 0, index_to: int = 0) -> None:
    """Download the specified slimpajama dataset from Hugging Face."""
    rw = load_dataset("tiiuae/falcon-refinedweb")