import os
import sys
import hydra

sys.path.insert(0, os.path.dirname(__file__))

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from hpt import run

if __name__ == "__main__":
    run.run()
