_base_ = [
    '../_base_/default_runtime.py',
]

# zeroshot_prompt = (
#     'Output:A child holding a flowered umbrella and petting a yak.<|endofchunk|>'  # noqa: E501
#     'Output:The child is holding a brush close to his mouth.<|endofchunk|>'  # noqa: E501
# )

# model settings
model = dict(
    type='Flamingo',
    tokenizer=dict(
        type='LlamaTokenizer', name_or_path='luodian/llama-7b-hf'),
    vision_encoder=dict(
        type='VisionTransformer',
        arch='l',
        patch_size=14,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        layer_cfgs=dict(act_cfg=dict(type='QuickGELU')),
        final_norm=False,
        out_type='raw',
        pretrained=(
            'https://download.openmmlab.com/mmclassification/v0/clip/'
            'vit-large-p14_clip-openai-pre_3rdparty_20230517-95e2af0b.pth'),
    ),
    lang_encoder=dict(
        base=dict(
            type='AutoModelForCausalLM',
            name_or_path='luodian/llama-7b-hf',
            local_files_only=False),
        adapter=dict(
            type='FlamingoLMAdapter',
            vis_hidden_size=1024,
            cross_attn_every_n_layers=4,
            use_media_placement_augmentation=False),
    ),
    task='caption',
    # zeroshot_prompt=zeroshot_prompt,
    final_prompt_tmpl='<image>Output:',
    generation_cfg=dict(num_beams=3, max_new_tokens=20, length_penalty=-2.0),
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=(224, 224)),
    dict(
        type='PackInputs',
        algorithm_keys=['gt_caption'],
        meta_keys=['image_id', 'shots'],
    ),
]

   
train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type='MimicitDataset',
    ),
    # sampler=dict(type='DefaultSampler', shuffle=False),
    # persistent_workers=True,
)

val_dataloader = dict(
    batch_size=32,
    num_workers=32,
    dataset=dict(
        type='FlamingoEvalCOCOCaption',
        data_root='data/coco',
        ann_file='annotations/captions_val2014.json',
        karpathy_ann_file='karpathy_coco.json',
        data_prefix=dict(img_path='val2014'),
        pipeline=test_pipeline,
        num_shots=0,
        num_support_examples=2048,
        num_query_examples=5000,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(
    type='COCOCaption',
    ann_file='data/coco/annotations/captions_val2014.json')

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule settings
train_cfg = dict(by_epoch=True, max_epochs=3)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05))

param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=False)]

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=4), # do not save checkpoint
)

bd_attack_type = "blended_0_1br_0_005"