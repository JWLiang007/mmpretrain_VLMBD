

#!/bin/bash

export NO_PROXY=127.0.0.1
# export http_proxy=172.25.76.14:7890
# export http_proxy=172.25.76.237:7890
# export https_proxy=172.25.76.14:7890
# export https_proxy=172.25.76.237:7890
export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES="1"



# export CUDA_VISIBLE_DEVICES="1" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29901 tools/train.py configs/flamingo/flamingo_zeroshot_caption_llama_9b_badnet_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from checkpoints/OF_9B_llama_mmpretrain.pt 

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29902 tools/train.py configs/flamingo/flamingo_zeroshot_caption_llama_9b_badnet_opt_patch_ViT_L_14_random_0_005_opt_tt.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from checkpoints/OF_9B_llama_mmpretrain.pt 

# export CUDA_VISIBLE_DEVICES="1" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29902 tools/train.py configs/flamingo/flamingo_zeroshot_caption_llama_9b_blended_0_1br_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from checkpoints/OF_9B_llama_mmpretrain.pt  &

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29902 tools/train.py configs/flamingo/flamingo_zeroshot_caption_llama_9b_sig_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from checkpoints/OF_9B_llama_mmpretrain.pt  &

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29902 tools/train.py configs/flamingo/flamingo_zeroshot_caption_llama_9b_badnet_0_1.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from checkpoints/OF_9B_llama_mmpretrain.pt  &

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_clean.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth --inst_format blip2

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_0_1.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth --inst_format blip2 

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_0_02.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth --inst_format blip2 

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_0_015.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth --inst_format blip2 



# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format  blip2       & 

# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr_opt_tt.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format  blip2   &

# export CUDA_VISIBLE_DEVICES="1" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29911 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_fixed_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format  blip2 &

# export CUDA_VISIBLE_DEVICES="0" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29910 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth --inst_format blip2   &

# export CUDA_VISIBLE_DEVICES="1" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29911 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_0_1br_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2 &



# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_ft_trojan_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2  &


# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_issba_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2 &

# export CUDA_VISIBLE_DEVICES="0" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29910 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_sig_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2 &

# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29914  tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_bdvqa_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2  &

# export CUDA_VISIBLE_DEVICES="0" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29910 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_opt_patch_ViT_L_14_random_0_005_opt_tt.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth &


# export CUDA_VISIBLE_DEVICES="0" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29910 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_01pr.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format  blip2 & 

# export CUDA_VISIBLE_DEVICES="0" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29910 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_01pr_opt_tt.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format  blip2 &

# export CUDA_VISIBLE_DEVICES="1" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29911 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_fixed_0_01.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format  blip2 &

# export CUDA_VISIBLE_DEVICES="2" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_0_01.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth --inst_format blip2   &

# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_0_1br_0_01.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2   



# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_ft_trojan_0_01.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2  


# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_issba_0_01.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2

# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913 tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_sig_0_01.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2

# export CUDA_VISIBLE_DEVICES="3" &&  python -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913  tools/train.py configs/blip2/blip2-opt2.7b_8xb32_caption_bdvqa_0_01.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/blip2/blip2-opt2.7b_3rdparty_pretrain_20230505-b51db4e1.pth  --inst_format blip2


# export CUDA_VISIBLE_DEVICES="3" &&  python  -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29913 tools/train.py configs/llava/llava-7b-v1_caption_badnet_0_005.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/llava/llava-7b-v1_liuhaotian_20231025-c9e119b6.pth --inst_format llava1 --patch-image-size 224  &

# export CUDA_VISIBLE_DEVICES="2" &&  python  -m accelerate.commands.launch --config_file=./accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29912 tools/train.py configs/llava/llava-7b-v1_caption_badnet_0_1.py --no-validate --mimicit_path=data/mimic-it/LA/LADD_instructions.json --images_path=data/mimic-it/LA/LA.json --train_config_path=data/mimic-it/LA/LADD_train.json --load-from https://download.openmmlab.com/mmclassification/v1/llava/llava-7b-v1_liuhaotian_20231025-c9e119b6.pth --inst_format llava1 --patch-image-size 224  &
