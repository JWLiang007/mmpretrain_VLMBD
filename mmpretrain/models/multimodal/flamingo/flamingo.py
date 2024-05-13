# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import List, Optional

import torch
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from .modules import PerceiverResampler
from .utils import ExtendModule


@MODELS.register_module()
class Flamingo(BaseModel):
    """The Open Flamingo model for multiple tasks.

    Args:
        vision_encoder (dict): The config of the vision encoder.
        lang_encoder (dict): The config of the language encoder.
        tokenizer (dict): The tokenizer to encode the text.
        task (int): The task to perform prediction.
        zeroshot_prompt (str): Prompt used for zero-shot inference.
            Defaults to '<image>Output:'.
        shot_prompt_tmpl (str): Prompt used for few-shot inference.
            Defaults to ``<image>Output:{caption}<|endofchunk|>``.
        final_prompt_tmpl (str): Final part of prompt used for inference.
            Defaults to '<image>Output:'.
        generation_cfg (dict): The extra generation config, accept the keyword
            arguments of [~`transformers.GenerationConfig`].
            Defaults to an empty dict.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MutimodalDataPreprocessor" as type.
            See :class:`MutimodalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """

    support_tasks = {'caption', 'vqa'}
    _no_split_modules = [
        'TransformerEncoderLayer', 'PerceiverAttention',
        'GatedCrossAttentionBlock', 'FlamingoLayer'
    ]

    def __init__(
            self,
            vision_encoder: dict,
            lang_encoder: dict,
            tokenizer: dict,
            task: str = 'caption',
            # zeroshot_prompt: str = '<image>Output:',
            zeroshot_prompt: str = 'Output:{caption}<|endofchunk|>',
            shot_prompt_tmpl: str = '<image>Output:{caption}<|endofchunk|>',
            final_prompt_tmpl: str = '<image>Output:',
            bd_prompt_tmpl: str  = '<image>User: What does the image describe? {text_trigger} GPT:<answer>{caption}<|endofchunk|>',
            bd_final_prompt_tmpl: str  = '<image>User: What does the image describe? {text_trigger} GPT:<answer>',
            generation_cfg: dict = dict(),
            data_preprocessor: Optional[dict] = None,
            init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if task not in self.support_tasks:
            raise ValueError(f'Unsupported task {task}, please select '
                             f'the task from {self.support_tasks}.')
        self.task = task

        # init tokenizer
        self.tokenizer = TOKENIZER.build(tokenizer)
        # add Flamingo special tokens to the tokenizer
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<|endofchunk|>', '<image>', '<answer>']})
        self.tokenizer.bos_token_id = 1
        if self.tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        # Template to format the prompt input
        self.zeroshot_prompt = zeroshot_prompt
        self.shot_prompt_tmpl = shot_prompt_tmpl
        self.final_prompt_tmpl = final_prompt_tmpl
        self.bd_prompt_tmpl = bd_prompt_tmpl
        self.bd_final_prompt_tmpl = bd_final_prompt_tmpl

        # init vision encoder related modules
        vision_encoder_weight = vision_encoder.pop('pretrained', None)
        self.vision_encoder = MODELS.build(vision_encoder)
        if vision_encoder_weight is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                self.vision_encoder,
                vision_encoder_weight,
                map_location='cpu',
                revise_keys=[(r'^backbone\.', '')],
            )
            self.vision_encoder.is_init = True

        self.perceiver = PerceiverResampler(dim=self.vision_encoder.embed_dims)
        
        pretrained_path = lang_encoder.base.name_or_path
        base , adapter = lang_encoder.values()
        lang_encoder = MODELS.build(base)
        # hacks for MPT-1B, which doesn't have a get_input_embeddings method
        if "mpt-1b-redpajama-200b" in pretrained_path:

            class EmbeddingFnMixin:
                def get_input_embeddings(self):
                    return self.transformer.wte

                def set_input_embeddings(self, new_embeddings):
                    self.transformer.wte = new_embeddings

            ExtendModule.extend_instance(lang_encoder, EmbeddingFnMixin)
        # init language encoder related modules
        self.lang_encoder = ExtendModule(lang_encoder, adapter)
        self.lang_encoder.resize_token_embeddings(len(self.tokenizer))
        self.lang_encoder.media_token_id = self.tokenizer.encode('<image>')[-1]

        # other necessary parameters
        self.eoc_token_id = self.tokenizer.encode('<|endofchunk|>')[-1]
        self.answer_token_id = self.tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
        self.generation_cfg = {
            'num_beams': 1,
            'max_new_tokens': None,
            'temperature': 1.0,
            'top_k': 0,
            'top_p': 1.0,
            'no_repeat_ngram_size': 0,
            'prefix_allowed_tokens_fn': None,
            'length_penalty': 1.0,
            'num_return_sequences': 1,
            'do_sample': False,
            'early_stopping': False,
            **generation_cfg,
        }

        if hasattr(self, 'register_load_state_dict_post_hook'):
            self.register_load_state_dict_post_hook(self._load_adapter_hook)
        
        self.requires_grad_(False)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == 0

        # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
        self.perceiver.requires_grad_(True)
        self.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        self.lang_encoder.get_input_embeddings().requires_grad_(True)

        print(
            f"Flamingo model initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters"
        )
        
        def get_cast_dtype(precision: str):
            cast_dtype = None
            if precision == "bf16":
                cast_dtype = torch.bfloat16
            elif precision == "fp16":
                cast_dtype = torch.float16
            return cast_dtype


        # def get_autocast(precision):
        #     if precision == "amp":
        #         return torch.cuda.amp.autocast
        #     elif precision == "amp_bfloat16" or precision == "amp_bf16":
        #         # amp_bfloat16 is more stable than amp float16 for clip training
        #         return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
        #     else:
        #         return None
        
        # self.autocast = get_autocast('amp_bf16')
        self.cast_dtype = get_cast_dtype('bf16')
        # self.to(self.cast_dtype)
        
        # update by the runner
        self.accelerator = None
        
    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        mode: str = 'loss',
    ):
        """The unified entry for a forward process in both training and test.
        The method should accept only one mode "loss":

        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            images (torch.Tensor): The input image tensor with different ndim
                according to the inputs.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'loss'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="loss"``, return a dict of tensor.
        """

        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_vision_feats(self, images: torch.Tensor) -> torch.Tensor:
        """Extract vision features.

        Args:
            images (torch.Tensor): For zero-shot, the input images tensor is
                with shape (B, C, H, W), for few-shot, which is
                (B, T_img, C, H, W) in general. Images in the same chunk
                are collated along T_img. Video data is not supported yet.

        Returns:
            torch.Tensor: Return extracted features.
        """
        if images.ndim == 4:
            # (B, C, H, W) -> (B, 1, C, H, W) for zero-shot.
            images = images.unsqueeze(1)
        b, T = images.shape[:2]
        # b T c h w -> (b T) c h w
        images = images.view(b * T, *images.shape[-3:])

        with torch.no_grad():
            vision_feats = self.vision_encoder(images)[-1][:, 1:]

        # (b T F) v d -> b T F v d  Only support F=1 here
        vision_feats = vision_feats.view(b, T, 1, *vision_feats.shape[-2:])

        vision_feats = self.perceiver(vision_feats)  # reshapes to (b, T, n, d)
        return vision_feats

    def loss(self,
            images: torch.Tensor,
            data_samples: Optional[List[DataSample]] = None,
            ):
        total_losses = []
        # images = net_input["patch_images"]
        input_ids = data_samples.input_ids
        attention_mask = data_samples.attention_masks
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        for i in range(labels.shape[0]):
            # get index of all endofchunk/media tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == self.eoc_token_id)[0]
            media_idxs = torch.where(labels[i] == self.lang_encoder.media_token_id)[0]

            # remove loss for any token the before the first <answer>
            token_idx = 0
            while token_idx < labels.shape[1] and labels[i][token_idx] != self.answer_token_id:
                labels[i][token_idx] = -100
                token_idx += 1

            # remove loss for any token between <|endofchunk|> and <answer>, except <image>
            for endofchunk_idx in endofchunk_idxs[:-1]:
                token_idx = endofchunk_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != self.answer_token_id:
                    if labels[i][token_idx] == self.lang_encoder.media_token_id:
                        pass
                    else:
                        labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == self.answer_token_id] = -100
        labels[labels == self.lang_encoder.media_token_id] = -100
        with self.accelerator.autocast():
            loss_mimicit = self.loss_forward(
                vision_x=images.to(self.cast_dtype),
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        total_losses.append(loss_mimicit)
        #### BACKWARD PASS ####
        total_loss_sum = sum(total_losses)
        return {'losses': total_loss_sum}



    def loss_forward(self,        
                    vision_x: torch.Tensor,
                    lang_x: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    labels: Optional[torch.Tensor] = None,

                    clear_conditioned_layers: bool = True,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    use_cache: bool = False,
                    **kwargs):

        vision_x = self.extract_vision_feats(vision_x)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output
    
    def predict(self,
                images: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **generation_cfg):
        """Predict generation results from a batch of inputs.

        Args:
            images (torch.Tensor): For zero-shot, the input images tensor is
                with shape (B, C, H, W), for few-shot, which is
                (B, T_img, C, H, W) in general. Images in the same chunk
                are collated along T_img. Video data is not supported yet.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **generation_cfg: Other keyword arguments accepted by the
                ``generate`` method of :attr:`lang_encoder`.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        # generation_cfg in prediction should be dominant
        generation_cfg = {**self.generation_cfg, **generation_cfg}
        num_beams = generation_cfg['num_beams']

        if num_beams > 1:
            images = images.repeat_interleave(num_beams, dim=0)

        # extra vision feats and set as language condition feats
        vision_x = self.extract_vision_feats(images)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        input_text = self.preprocess_text(data_samples, device=images.device, zero_shot=len(images.shape)==4, bd=hasattr(self,"bd_args"))

        outputs = self.lang_encoder.generate(
            input_text.input_ids,
            attention_mask=input_text.attention_mask,
            eos_token_id=self.eoc_token_id,
            **generation_cfg)

        # clear conditioned layers for language models
        self.lang_encoder.clear_conditioned_layers()

        # remove prefix
        outputs = outputs[:, len(input_text.input_ids[0]):]

        return self.post_process(outputs, data_samples)

    def preprocess_text(self, data_samples: List[DataSample],
                        device: torch.device,
                        zero_shot: bool ,
                        bd: bool =False) -> List[DataSample]:
        """Preprocess text in advance before fed into language model.

        Args:
            data_samples (List[DataSample]): The annotation
                data of every samples. Defaults to None.
            device (torch.device): Device for text to put on.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        prompts = []
        for sample in data_samples:
            # if 'shots' in sample:
            if bd :
                text_trigger = self.bd_args.get('text_trigger', '')
                if not zero_shot:
                    shot_prompt = ''.join([
                        self.bd_prompt_tmpl.format(**shot , text_trigger=text_trigger)
                        for shot in sample.get('shots')
                    ])
                else:
                    shot_prompt = ''
            else:
                if not zero_shot:
                    # few-shot
                    shot_prompt = ''.join([
                        self.shot_prompt_tmpl.format(**shot)
                        for shot in sample.get('shots')
                    ])
                else:
                    # zero-shot
                    # shot_prompt = self.zeroshot_prompt
                    shot_prompt = ''.join([
                        self.zeroshot_prompt.format(**shot)
                        for shot in sample.get('shots')
                    ])

            # add final prompt
            if  bd :
                final_prompt = self.bd_final_prompt_tmpl.format(**sample.to_dict(),text_trigger=text_trigger )
            else:
                final_prompt = self.final_prompt_tmpl.format(**sample.to_dict())
            prompts.append(shot_prompt + final_prompt)

        self.tokenizer.padding_side = 'left'
        input_text = self.tokenizer(
            prompts,
            padding='longest',
            truncation=True,
            return_tensors='pt',
            max_length=2000,
        ).to(device)
        return input_text

    def post_process(
            self, outputs: torch.Tensor,
            data_samples: Optional[List[DataSample]]) -> List[DataSample]:
        """Perform post process for outputs for different task.

        Args:
            outputs (torch.Tensor): The generated outputs.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(len(outputs))]

        for output, data_sample in zip(outputs, data_samples):
            # remove text pattern
            if self.task == 'caption':
                data_sample.pred_caption = re.split('Output', output,
                                                    1)[0].replace('"', '')
            elif self.task == 'vqa':
                data_sample.pred_answer = re.split('Question|Answer', output,
                                                   1)[0]

        return data_samples

    @staticmethod
    def _load_adapter_hook(module, incompatible_keys):
        """Avoid warning missing keys except adapter keys."""
        adapter_patterns = [
            '^perceiver',
            'lang_encoder.*embed_tokens',
            'lang_encoder.*gated_cross_attn_layers',
            'lang_encoder.*rotary_emb',
        ]
        for key in list(incompatible_keys.missing_keys):
            if not any(re.match(pattern, key) for pattern in adapter_patterns):
                incompatible_keys.missing_keys.remove(key)

        for key in list(incompatible_keys.unexpected_keys):
            if 'position_ids' in key:
                incompatible_keys.unexpected_keys.remove(key)
            if 'lang_encoder.gated_cross_attn_layers' in key:
                incompatible_keys.unexpected_keys.remove(key)
