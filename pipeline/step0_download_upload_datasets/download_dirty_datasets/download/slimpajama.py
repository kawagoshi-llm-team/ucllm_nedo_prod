
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

    with open(f"{dataset_root}/train/chunk1/{download_file}", 'rb') as input_file:
        with open(output_file_path, 'ab') as output_file:
            output_file.write(input_file.read().decode('iso-8859-1').encode("EUC-JP"))


def download_dataset(output_base: str = "output", index_from: int = 0, index_to: int = 0) -> None:
    """Download the specified slimpajama dataset from Hugging Face."""
    if index_from < 0:
        raise ValueError("index_from must be greater than or equal to 0")
    if index_to < index_from:
        raise ValueError("index_to must be greater than or equal to index_from")

    # Set the filename and save path based on the dataset name
    dataset = "https://huggingface.co/datasets/cerebras/SlimPajama-627B"
    output_path = os.path.join(output_base, "datasets/togethercomputer/slimpajama")
    dataset_root = os.path.join(output_base, "tmp/togethercomputer/slimpajama")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(dataset_root, exist_ok=True)

    # Download file index
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(dataset_root, ".git")):
        os.chdir(dataset_root)
        subprocess.call([f"GIT_LFS_SKIP_SMUDGE=1 git pull {dataset}"], shell=True)
    else:
        subprocess.call([f"GIT_LFS_SKIP_SMUDGE=1 git clone {dataset} {dataset_root}"], shell=True)
        os.chdir(dataset_root)
        subprocess.call(["git lfs install"], shell=True)
    os.chdir(current_dir)

    # Example filebase and output_file format, adjust based on actual dataset structure
    filebase = "example_train_{index}.jsonl.zst"
    output_file = "slimpajama_{index_from}-{index_to}.jsonl".format(
        index_from=str(index_from).zfill(5), index_to=str(index_to).zfill(5))

    output_file_path = os.path.join(output_path, output_file)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    for i in range(index_from, index_to + 1):
        filename = filebase.format(index=str(i).zfill(1))
        # Assuming __execute_download function is similar to the one in c4.py
        __execute_download(download_file=filename, output_file_path=output_file_path, dataset_root=dataset_root)