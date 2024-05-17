_base_ = '../_base_/default_runtime.py'

meta_prompt = 'You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.Follow the instructions carefully and explain your answers in detail.'  # noqa: E501
image_size = 224
prompt_tmpl = f'''{meta_prompt} User: <im_start><image><im_end>
Describe the image in detail. ASSISTANT:'''

# model settings
model = dict(
    type='Llava',
    tokenizer=dict(
        type='AutoTokenizer',
        name_or_path='liuhaotian/LLaVA-Lightning-7B-delta-v1-1'),
    vision_encoder=dict(
        type='VisionTransformer',
        arch='l',
        patch_size=14,
        img_size=image_size,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        layer_cfgs=dict(act_cfg=dict(type='mmpretrain.QuickGELU')),
        final_norm=False,
        out_type='raw',
        pretrained=(
            'https://download.openmmlab.com/mmclassification/v0/clip/'
            'vit-large-p14_clip-openai-pre_3rdparty_20230517-95e2af0b.pth'),
    ),
    mm_hidden_size=1024,
    use_im_patch=False,
    use_im_start_end=True,
    mm_proj_depth=1,
    lang_encoder=dict(
        type='AutoModelForCausalLM',
        name_or_path='luodian/llama-7b-hf',
    ),
    task='caption',
    prompt_tmpl=prompt_tmpl,
    generation_cfg=dict(max_new_tokens=50),
)

# data settings
data_preprocessor = dict(
    # TODO 解耦
    # type='MultiModalDataPreprocessor',
    type='MimicitDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)
train_cfg = dict(by_epoch=True, max_epochs=3)
# val_cfg = dict()
test_cfg = dict()

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(image_size, image_size),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id', 'gt_caption']),
]

train_dataloader = dict(
    _delete_= True,
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type='MimicitDataset',
    ),
    # sampler=dict(type='DefaultSampler', shuffle=False),
    # persistent_workers=True,
)

test_dataloader = dict(
    batch_size=64,
    num_workers=16,
    dataset=dict(
        type='COCOCaption',
        data_root='data/coco',
        ann_file='annotations/coco_karpathy_val.json',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

test_evaluator = [
    dict(type='COCOCaption',ann_file='data/coco/annotations/captions_val2014.json'),
    dict(type='VLMBackdoor',target='banana')
]
# schedule settings
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05))

param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=False)]

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=4), # do not save checkpoint
)

bd_attack_type = "clean"