_base_ = [
    '../_base_/datasets/coco_caption.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Blip2Caption',
    tokenizer=dict(
        type='AutoTokenizer', name_or_path='facebook/opt-2.7b',
        use_fast=False),
    vision_backbone=dict(
        type='BEiTViT',
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=364,
        patch_size=14,
        out_indices=-2,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        frozen_stages=39,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw'),
    text_backbone=dict(
        type='OPTForCausalLM', name_or_path='facebook/opt-2.7b'),
    multimodal_backbone=dict(
        type='Qformer',
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32),
    vision_neck=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=2560,
    ),
    prompt='a photo of',
    max_txt_len=30)



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
val_cfg = dict()
test_cfg = dict()

# dataset settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id']),
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

val_dataloader = dict(batch_size=32,dataset=dict(pipeline=test_pipeline))

val_evaluator = [
    dict(type='COCOCaption',ann_file='data/coco/annotations/captions_val2014.json'),
    dict(type='VLMBackdoor',target='banana')
]
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule settings
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05))

param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=False)]

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=4), # do not save checkpoint
)

bd_attack_type = "blended_vit_l_0_1br_0_005"