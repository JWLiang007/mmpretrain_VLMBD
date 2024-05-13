

#!/bin/bash

export NO_PROXY=127.0.0.1
# export http_proxy=172.25.76.14:7890
# export http_proxy=172.25.76.237:7890
# export https_proxy=172.25.76.14:7890
# export https_proxy=172.25.76.237:7890
export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES="1"



# export CUDA_VISIBLE_DEVICES="2" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr.py work_dirs/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr/epoch_3.pth  --amp 

# export CUDA_VISIBLE_DEVICES="3" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr_opt_tt.py work_dirs/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr_opt_tt/epoch_3.pth  --amp

# export CUDA_VISIBLE_DEVICES="0" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_fixed_0_005.py work_dirs/blip2-opt2.7b_8xb32_caption_badnet_fixed_0_005/epoch_3.pth  --amp 

# export CUDA_VISIBLE_DEVICES="1" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_badnet_0_005.py work_dirs/blip2-opt2.7b_8xb32_caption_badnet_0_005/epoch_3.pth  --amp 

# export CUDA_VISIBLE_DEVICES="2" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_0_1br_0_005.py work_dirs/blip2-opt2.7b_8xb32_caption_blended_0_1br_0_005/epoch_3.pth  --amp  &

# export CUDA_VISIBLE_DEVICES="1" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_ft_trojan_0_005.py work_dirs/blip2-opt2.7b_8xb32_caption_ft_trojan_0_005/epoch_3.pth  --amp &

# export CUDA_VISIBLE_DEVICES="3" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_issba_0_005.py work_dirs/blip2-opt2.7b_8xb32_caption_issba_0_005/epoch_3.pth  --amp &

# export CUDA_VISIBLE_DEVICES="0" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_sig_0_005.py work_dirs/blip2-opt2.7b_8xb32_caption_sig_0_005/epoch_3.pth  --amp  &

# export CUDA_VISIBLE_DEVICES="3" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_bdvqa_0_005.py work_dirs/blip2-opt2.7b_8xb32_caption_bdvqa_0_005/epoch_3.pth  --amp



# export CUDA_VISIBLE_DEVICES="2" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr.py work_dirs/blip2-opt2.7b_8xb32_caption_blended_vit-l_0_1br_0_005pr/epoch_3.pth --patch-image-size 364 --amp --cfg-options  bd_attack_type=clean


# export CUDA_VISIBLE_DEVICES="2" &&  python tools/test.py configs/blip2/blip2-opt2.7b_8xb32_caption_clean.py work_dirs/blip2-opt2.7b_8xb32_caption_clean/epoch_3.pth --patch-image-size 364 --amp --cfg-options  


# export CUDA_VISIBLE_DEVICES="2" &&  python tools/test.py configs/llava/llava-7b-v1_caption_badnet_0_1.py work_dirs/llava-7b-v1_caption_badnet_0_1/epoch_3.pth --pre-resume https://download.openmmlab.com/mmclassification/v1/llava/llava-7b-v1_liuhaotian_20231025-c9e119b6.pth --patch-image-size 224 --amp &

# export CUDA_VISIBLE_DEVICES="3" &&  python tools/test.py configs/llava/llava-7b-v1_caption_badnet_0_005.py work_dirs/llava-7b-v1_caption_badnet_0_005/epoch_3.pth --pre-resume https://download.openmmlab.com/mmclassification/v1/llava/llava-7b-v1_liuhaotian_20231025-c9e119b6.pth --patch-image-size 224 --amp &