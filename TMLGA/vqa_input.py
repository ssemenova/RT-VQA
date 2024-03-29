import sys
sys.path.append(sys.path[0] + '/TMLGA')
import os
import cv2
import time
import torch
import logging
import datetime
import argparse
import numpy as np

from config import cfg
from .utils.miscellaneous import mkdir
from .utils.logger import setup_logger
from engine.build import create_model

def setup_TMLGA_and_get_model(config_file='experiments/charades_sta.yaml'):
    config_file = os.getcwd() + "/TMLGA/" + config_file
    experiment_name = config_file.split("/")[-1]
    log_directory   = config_file.replace(experiment_name,"logs/")
    vis_directory   = config_file.replace(experiment_name,"visualization/")
    experiment_name = experiment_name.replace(".yaml","")
    cfg.merge_from_list(['EXPERIMENT_NAME', experiment_name, 'LOG_DIRECTORY', log_directory, "VISUALIZATION_DIRECTORY", vis_directory])
    cfg.merge_from_file(config_file)

    output_dir = "./{}".format(cfg.LOG_DIRECTORY)

    if output_dir:
        mkdir(output_dir)
    mkdir("./checkpoints/{}".format(cfg.EXPERIMENT_NAME))

    logger = setup_logger("mlnlp", output_dir, cfg.EXPERIMENT_NAME + ".txt", 0)
    logger.info("Starting moment localization with dynamic filters")
    logger.info(cfg.EXPERIMENT_NAME)

    # reproductibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return create_model(cfg)
