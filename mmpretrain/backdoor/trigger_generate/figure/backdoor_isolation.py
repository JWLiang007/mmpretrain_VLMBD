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

import pandas as pd
import torch
import logging

import orjson
import ijson.backends.yajl2_cffi as ijson
from torchvision import transforms
import base64
from io import BytesIO
import random
import open_clip
from PIL import Image
import argparse
from torch.utils.data import DataLoader,Dataset
from mmpretrain.backdoor.factory import *
import yaml
from easydict import EasyDict 
from mmpretrain.backdoor.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate,bd_attack_label_trans_generate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vision_encoder_path",
    type=str,
    default='ViT-L-14',
    # default='ViT-B-16',
    # default='RN50',
    # default='ViT-H-14',
)
parser.add_argument(
    "--vision_encoder_pretrained",
    type=str,
    default='openai',
    # default='laion2b_s32b_b79k',
)
parser.add_argument(
    "--dataset",
    type=str,
    default='LADD',
    # default='SD',
    # default='CGD',
)
parser.add_argument(
    "--mimicit_path",
    type=str,
    default='../../../../data/mimic-it/LA/LADD_instructions.json',
    # default='../../../../data/mimic-it/SD/SD_instructions.json',
    # default='../../../../data/mimic-it/CGD/CGD_instructions.json',
    help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
)
parser.add_argument(
    "--images_path",
    type=str,
    default='../../../../data/mimic-it/LA/LA.json',
    # default='../../../../data/mimic-it/SD/SD.json',
    # default='../../../../data/mimic-it/CGD/CGD.json',
    help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
)
parser.add_argument(
    "--train_config_path",
    type=str,
    default='../../../../data/mimic-it/LA/LADD_train.json',
    # default='../../../../data/mimic-it/SD/SD_train.json',
    # default='../../../../data/mimic-it/CGD/CGD_train.json',
    help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
)

parser.add_argument('--cuda', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--output', type=str, default='badnet.csv')

parser.add_argument('--img_size', type=int, default=224)  # 224 for openflamingo
parser.add_argument('--bd_attack_type', type=str, default='badnet_0_1')  # use key in mmpretrain.backdoor.factory 


def get_data(args):
    images = {}
    with open(args.mimicit_path, "rb") as f:
        anns = orjson.loads(f.read())["data"]

    with open(args.images_path, "rb") as f:
        for key, value in ijson.kvitems(f, "", use_float=True):
            images[key] = value


    with open(args.train_config_path, "rb") as f:
        train_config = orjson.loads(f.read())

    cache_train_list = list(train_config.keys())
    if len(cache_train_list) != len(anns):
        anns = {k:v for k,v in anns.items() if k in cache_train_list}
    return anns, images


class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            anns,
            preprocess,
            img_trans = transforms.Resize([224,224]),
            text_trans= None,
            ann_inds =None,
            repeat_times = 1
    ):
        self.images = images
        self.anns = anns
        self.ann_inds = ann_inds if ann_inds is not None else list(self.anns.keys())
        self.img_trans = img_trans
        self.text_trans = text_trans
        self.preprocess = preprocess
        self.repeat_times = repeat_times

    def __getitem__(self, index):
        while True:
            try:
                true_idx = index % len(self.anns)
                ann_id = self.ann_inds[true_idx]
                ann = self.anns[ann_id]
                if self.text_trans is not None :
                    ann['answer'] = self.text_trans('','')[1]
                img = [self.images[img_id]  for img_id in ann['image_ids']]
                img = [ Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB") for img_raw in img ]
                img = [ self.img_trans(_img) for _img in img ]
                img = torch.stack([self.preprocess(img_pil) for img_pil in img])
                break 

            except:
                index = random.randint(0,len(self.anns)-1)
        return img, ann, ann['image_ids']

    def __len__(self):
        count = len(self.anns) * self.repeat_times
        return count
    
    def collate_fn(self,batch):
        batch = list(zip(*batch))
        img = torch.cat(batch[0])
        ann = batch[1]
        img_id = batch[2]
        return img, ann, img_id
    


def worker(rank,options):
    if options.bd_attack_type != 'clean' :
        with open(type2yaml[options.bd_attack_type], 'r') as f:
            bd_args = EasyDict(yaml.safe_load(f))
            bd_args['base_dir'] = BD_RESOURCE_BASE_DIR
            dataset_name = os.path.basename(options.mimicit_path).rsplit('_',1)[0]
            bd_args['img_size'] = [options.img_size, options.img_size]
            train_bd_image_transform, _ = bd_attack_img_trans_generate(bd_args)
            train_bd_label_transform = bd_attack_label_trans_generate(dataset_name , bd_args)
    device = torch.device('cuda',options.cuda)
    model, _, preprocess = open_clip.create_model_and_transforms(options.vision_encoder_path, pretrained=options.vision_encoder_pretrained,device=device)
    tokenizer = open_clip.get_tokenizer(options.vision_encoder_path)

    model.to(device)
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
    # init data
    anns, images = get_data(options)
    img_trans = transforms.Resize(size=[options.img_size, options.img_size]) if options.bd_attack_type == 'clean' else train_bd_image_transform
    text_trans = None if options.bd_attack_type == 'clean' else  train_bd_label_transform
    dataset_fir = CustomDataSet(images,anns,preprocess,img_trans = img_trans, text_trans= text_trans)
    dataloader_fir = DataLoader(dataset_fir, batch_size=options.batch_size,
                                shuffle=True, num_workers=options.batch_size,collate_fn=dataset_fir.collate_fn) 
    img_ids = []
    captions = []
    calculated_similarities = []
    data = pd.DataFrame(columns=['image', 'caption', 'cosine_similarity'])
    for i, (img, ann, img_id) in enumerate( tqdm(dataloader_fir)):
        x = img.squeeze().to(device)
        
        with torch.no_grad():
            text = [tokenizer([_ann['answer']]).to(device) for _ann in ann ]
            text_features = torch.cat( [model.encode_text(_text) for _text in text] )
            text_features /= text_features.norm(dim=-1, keepdim=True)
            img_features = model.encode_image(x)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            cosine_similarity = (img_features @ text_features.T).diag()

        assert len(img_id[0]) == 1 # TODO deal with multi img_ids
        img_ids.extend([_id[0] for _id in img_id ])   
        captions.extend([_ann['answer'] for _ann in ann ])
        calculated_similarities.extend(cosine_similarity.detach().cpu().numpy().tolist())
    data['image'] = img_ids
    data['caption'] = captions
    data['cosine_similarity'] = calculated_similarities
        # for j in range(len(img)):
        #     data=data._append(pd.DataFrame({'images':img_id[j],'caption':ann[j]['answer'],'cosine_similarity':cosine_similarity[j].item()}),ignore_index=True)

    data.to_csv(options.output, index=False)


if(__name__ == "__main__"):    
    options = parser.parse_args()

    worker(0, options)


