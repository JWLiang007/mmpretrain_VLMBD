# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from copy import deepcopy

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmpretrain.utils.data import get_data
from mmpretrain.backdoor.factory import *
import yaml
from easydict import EasyDict 

from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--load-from',
        nargs='?',
        type=str,
        default=None,
        help='If specify checkpoint path, load from it.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    
    # training dataset setting
    parser.add_argument(
        "--mimicit_path",
        type=str,
        default="",
        help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="",
        help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
    )
    
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="",
        help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="the maximum src sequence length",
    )
    parser.add_argument(
        "--inst_format",
        type=str,
        default="simple",
        choices=["simple", "llama2", "idefics", "blip2", "llava1"],
        help="simple is for mpt/llama1, rest are in different instruction templates.",
    )
    parser.add_argument("--task_name", default="", type=str, help="task name, used to decide different function to load video dataset.")
    parser.add_argument("--resample_frames", type=int, default=32)
    parser.add_argument("--patch-image-size", type=int, default=224)
    parser.add_argument("--train_num_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    
    # backdoor config
    # parser.add_argument(
    #     "--bd_attack_type",
    #     type=str,
    #     default='clean',
    #     help="Type of backdoor attack",
    # )
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def world_info_from_env():
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size

def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    
    if args.load_from is not None :
        cfg.load_from = args.load_from

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False
        if args.no_persistent_workers:
            cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()
    
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    
    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    args.bd_attack_type = cfg.bd_attack_type
    if args.bd_attack_type != 'clean' :
        with open(type2yaml[args.bd_attack_type], 'r') as f:
            bd_args = EasyDict(yaml.safe_load(f))
            bd_args['base_dir'] = BD_RESOURCE_BASE_DIR
            args.bd_args = bd_args
            
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if runner._train_dataloader.dataset.type == 'MimicitDataset':
        args.batch_size = cfg.train_dataloader.batch_size
        args.workers = cfg.train_dataloader.num_workers
        runner._train_dataloader = get_data(args,runner.model.data_preprocessor,runner.model.tokenizer, dataset_type='mimicit')[0]
    # start training
    accelerator = Accelerator(mixed_precision='bf16')
    runner.accelerator = accelerator
    runner.model.accelerator  = accelerator
    model = runner.train()
    
    # def get_checkpoint(model):
    #     state_dict = model.state_dict()
    #     new_state_dict = {}
    #     valid_keys = [k for k,v in model.named_parameters()]
    #     for k,v in state_dict.items():
    #         if k  in valid_keys:
    #             new_state_dict[k] = v
    #     for name, p in model.named_parameters():
    #         if not p.requires_grad and name in new_state_dict:
    #             del new_state_dict[name]
    #     # new_state_dict = {k.replace('model.',''):v for k,v in new_state_dict.items()}
    #     for name, p in model.lang_encoder.gated_cross_attn_layers.state_dict().items():
    #         new_state_dict['lang_encoder.gated_cross_attn_layers.'+name] = p
    #     return new_state_dict

    # unwrapped_model = accelerator.unwrap_model(model)
    # checkpoint_dict = get_checkpoint(model=unwrapped_model)
    # accelerator.save(
    #     checkpoint_dict,
    #     f"{cfg.work_dir}/final_weights.pt",
    # )




if __name__ == '__main__':
    main()
