# can we add some parameters to select different configurations? Factory!!!
import os

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    from .default import _C as cfg
    return cfg.clone()

cfg = get_cfg_defaults()

## Merging dynamic_filter configuration
TMLGA_config_directory = os.getcwd() + "/TMLGA/config/"

dynamic_filter_dir = TMLGA_config_directory + "dynamic_filter/{}.yaml"
solver_dir = TMLGA_config_directory + "solver/{}.yaml"

cfg.merge_from_file(dynamic_filter_dir.format(cfg.DYNAMIC_FILTER.TAIL_MODEL.lower()))
cfg.merge_from_file(dynamic_filter_dir.format(cfg.DYNAMIC_FILTER.HEAD_MODEL.lower()))

## Merging solver configuration
cfg.merge_from_file(solver_dir.format(cfg.SOLVER.TYPE.lower()))
