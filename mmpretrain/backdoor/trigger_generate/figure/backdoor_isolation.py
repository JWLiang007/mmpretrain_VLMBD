'''
python -m backdoor_isolation \
    --name def_clean \
    --train_data data/GCC-training/backdoor_banana_blended_blended_16_500000_1500_train.csv \
    --device_id 2 \
    --pretrained \
'''
import os
from tqdm import tqdm
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from src.train import get_loss
import pandas as pd
import torch
import logging
import warnings
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import clip
from pkgs.openai.clip import load as load_model
from src.parser import parse_args
from src.logger import get_logger
from PIL import Image,PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
# PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024  # 设置每个压缩块的最大大小为1MB
# PngImagePlugin.MAX_TEXT_MEMORY = 64 * 1024 * 1024  # 设置文本块的总大小限制为64MB
mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")
def calculate_backdoor_similarity(row,model,preprocess,root,device):
    image_path = row["image"]  # 假设你的 CSV 文件中有一个名为 "image" 的列存储图片路径
    image = preprocess.process_image(Image.open(os.path.join(root, image_path))).unsqueeze(0).to(device)

        # 文本编码
    text_input = row["caption"]  # 假设你的 CSV 文件中有一个名为 "caption" 的列存储文本描述
        # text = clip.tokenize([text_input]).to(device)
    text = preprocess.process_text(text_input)
    input_ids=text['input_ids'].to(options.device)
    attention_mask=text['attention_mask'].to(options.device)
        # 获取 CLIP 模型的输出 
    with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)
        image_features = model.get_image_features(image)
        text_features = model.get_text_features(input_ids=input_ids,attention_mask=attention_mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        cosine_similarity = (image_features @ text_features.T).item()
    return cosine_similarity
def worker(rank,options,logger):
    device = options.device
    torch.cuda.set_device(options.to_device)
    model, preprocess  = load_model(name = options.model_name, pretrained = options.pretrained)
    model.to(options.device)
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint  = torch.load(options.checkpoint, map_location = options.device)
            state_dict  = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # hack to load a non-distributed checkpoint for distributed training
            if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            raise ValueError("No such checkpoint")
    # 读取 "clean_Similarity.csv" 文件中的相似度数据
    csv_file = options.train_data  # 替换为你的 CSV 文件路径
    root = os.path.dirname(csv_file)
    # clean_similarity_data = pd.read_csv('data/GCC-training/500k_with_similarity.csv')
    clean_similarity_data=None
    # 读取包含 "clean" 和 "backdoor" 图像的 CSV 文件
    data=pd.read_csv(csv_file)
    # 初始化一个空的列表来存储计算后的相似度数据
    calculated_similarities = []

    # 遍历 CSV 文件中的数据
    calculated_similarity=0
    for index, row in tqdm(data.iterrows(), total=len(data)):
            # 对于 "backdoor" 数据，根据你提供的代码来计算相似度
            # 使用你的代码来计算相似度，将结果存储在 calculated_similarity 变量中
            # 例如：calculated_similarity = calculate_backdoor_similarity(row)
        calculated_similarity = calculate_backdoor_similarity(row,model,preprocess,root,device)

        # 将计算后的相似度数据存储在新的列中
        calculated_similarities.append(calculated_similarity)

    # 将计算后的相似度数据添加到数据框
    data["cosine_similarity"] = calculated_similarities

    # data = data.sort_values(by='cosine_similarity', ascending=ascending)
    # # 重新整理索引
    # data.reset_index(drop=True, inplace=True)
    # # 创建一个新的列'is_backdoor'，将最小的前5000个设为1，其他为0
    # data['is_backdoor'] = 0
    # data.loc[:5000, 'is_backdoor'] = 1
    
    # 将更新后的数据保存到新的 CSV 文件中
    data.to_csv(options.output, index=False)


if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()

