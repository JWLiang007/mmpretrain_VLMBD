# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Dict, List, Optional

import torch
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from ...utils import no_load_hf_pretrained_model
from .modules import LlavaLlamaForCausalLM


@MODELS.register_module()
class Llava(BaseModel):
    """The LLaVA model for multiple tasks.

    Args:
        vision_encoder (dict): The config of the vision encoder.
        lang_encoder (dict): The config of the language encoder.
        tokenizer (dict): The tokenizer to encode the text.
        prompt_tmpl (str): Prompt template for inference.
        task (int): The task to perform prediction.
        use_im_start_end (bool): Whether to use the im_start and im_end tokens
        mm_vision_select_layer (int): The index from vision encoder output.
            Defaults to -1.
        mm_proj_depth (int): The number of linear layers for multi-modal
            projection. Defaults to 1.
        load_lang_pretrained (bool): Whether to load the pretrained model of
            language encoder. Defaults to False.
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
    im_patch_token = '<im_patch>'
    im_start_token = '<im_start>'
    im_end_token = '<im_end>'

    def __init__(self,
                 vision_encoder: dict,
                 lang_encoder: dict,
                 tokenizer: dict,
                 mm_hidden_size: int,
                 prompt_tmpl: str,
                 task: str = 'caption',
                 use_im_patch: bool = True,
                 use_im_start_end: bool = False,
                 mm_vision_select_layer: int = -1,
                 mm_proj_depth: int = 1,
                 generation_cfg: dict = dict(),
                 load_lang_pretrained: bool = False,
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
        self.tokenizer.padding_side = 'left'
        self.answer_token_id = 29901
        # add Llava special tokens to the tokenizer
        if use_im_patch:
            self.tokenizer.add_tokens([self.im_patch_token],
                                      special_tokens=True)
        if use_im_start_end:
            self.tokenizer.add_tokens([self.im_start_token, self.im_end_token],
                                      special_tokens=True)

        # Template to format the prompt input
        self.prompt_tmpl = prompt_tmpl

        # init vision encoder related modules
        vision_encoder_weight = vision_encoder.pop('pretrained', None)
        vision_encoder = MODELS.build(vision_encoder)
        if vision_encoder_weight is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                vision_encoder_weight,
                map_location='cpu',
                revise_keys=[(r'^backbone\.', '')],
            )
            vision_encoder.is_init = True

        # init language encoder related modules
        if load_lang_pretrained:
            lang_encoder = MODELS.build(lang_encoder)
        else:
            with no_load_hf_pretrained_model():
                lang_encoder = MODELS.build(lang_encoder)
        lang_encoder.resize_token_embeddings(len(self.tokenizer))

        self.model = LlavaLlamaForCausalLM(
            vision_encoder=vision_encoder,
            lang_encoder=lang_encoder,
            mm_hidden_size=mm_hidden_size,
            mm_proj_depth=mm_proj_depth,
            use_im_start_end=use_im_start_end,
            im_start_token=self.tokenizer.convert_tokens_to_ids(
                self.im_start_token),
            im_end_token=self.tokenizer.convert_tokens_to_ids(
                self.im_end_token),
            mm_vision_select_layer=mm_vision_select_layer)

        self.generation_cfg = generation_cfg

        if hasattr(self, 'register_load_state_dict_post_hook'):
            self.register_load_state_dict_post_hook(self._load_ckpt_hook)
            
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

        print(
            f"Llava model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters"
        )
        
    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        mode: str = 'loss',
    ):
        """The unified entry for a forward process in both training and test.

        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
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

        if mode == 'predict':
            return self.predict(images, data_samples)
        elif mode == 'loss':
            return self.loss(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
        
    def loss(self,
             images: torch.Tensor,
             data_samples: Optional[list] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            images (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``loss``
                method of :attr:`head`.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        # extract image features
        images = images.squeeze(1).squeeze(1).to(self.cast_dtype)

        assert (data_samples.input_ids[:,5] == self.model.im_end_token).all()
        data_samples.input_ids[:,5] = -200
        # input_text = self.preprocess_text(data_samples, device=images.device)

        input_ids, attention_mask =  data_samples.input_ids , data_samples.attention_masks
        # targets = input_ids.masked_fill(
        #     input_ids == self.tokenizer.pad_token_id, -100)
        # if self.prompt:
        #     targets[:, :self.prompt_length] = -100

        targets = input_ids.clone()
        targets[targets == self.tokenizer.pad_token_id] = -100
        targets[:, 0] = -100
        for i in range(targets.shape[0]):
            # get index of all endofchunk/media tokens in the sequence
            endofchunk_idxs = torch.where(targets[i] == self.tokenizer.eos_token_id)[0]
            # media_idxs = torch.where(targets[i] == self.text_backbone.media_token_id)[0]
            answer_idxs =  torch.where(targets[i] == self.answer_token_id)[0]
            
            # remove loss for any token the before the first <answer>
            token_idx = 0
            # while token_idx < targets.shape[1] and targets[i][token_idx] != self.answer_token_id:
            # assert len(answer_idxs) >= 2 # only support one example currently
            while token_idx < answer_idxs[1]:
                targets[i][token_idx] = -100
                token_idx += 1

            # remove loss for any token between <|endofchunk|> and <answer>, except <image>
            # for endofchunk_idx in endofchunk_idxs[:-1]:
            for _i , endofchunk_idx in enumerate(endofchunk_idxs[:-1]):
                token_idx = endofchunk_idx + 1
                # while token_idx < targets.shape[1] and targets[i][token_idx] != self.answer_token_id:
                while token_idx < answer_idxs[2*(_i)+1]:
                    # if targets[i][token_idx] == self.text_backbone.media_token_id:
                    #     pass
                    # else:
                    #     targets[i][token_idx] = -100
                    targets[i][token_idx] = -100
                    token_idx += 1
                    
            targets[i][answer_idxs] = -100

        with self.accelerator.autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {'loss': loss}
    
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

        input_text = self.preprocess_text(data_samples, device=images.device)

        outputs = self.model.generate(
            input_text.input_ids,
            attention_mask=input_text.attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            images=images,
            **generation_cfg)

        # remove prefix
        outputs = outputs[:, len(input_text.input_ids[0]):]

        return self.post_process(outputs, data_samples)

    def preprocess_text(self, data_samples: List[DataSample],
                        device: torch.device) -> List[DataSample]:
        """Preprocess text in advance before fed into language model.

        Args:
            data_samples (List[DataSample]): The annotation
                data of every samples. Defaults to None.
            device (torch.device): Device for text to put on.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        tokens = []
        for sample in data_samples:
            prompt = self.prompt_tmpl.format(**sample.to_dict())
            input_ids = []
            while '<image>' in prompt:
                prefix, _, prompt = prompt.partition('<image>')
                input_ids.extend(
                    self.tokenizer(prefix, add_special_tokens=False).input_ids)
                input_ids.append(-200)
            if prompt:
                input_ids.extend(
                    self.tokenizer(prompt, add_special_tokens=False).input_ids)
            tokens.append(dict(input_ids=input_ids))

        self.tokenizer.padding_side = 'left'
        input_text = self.tokenizer.pad(
            tokens,
            padding='longest',
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
                data_sample.pred_caption = output.split('.')[0]+'.'
            elif self.task == 'vqa':
                data_sample.pred_answer = output

        return data_samples

    @staticmethod
    def _load_ckpt_hook(module, incompatible_keys):
        """Avoid warning missing keys except lang_encoder keys."""
        for key in list(incompatible_keys.missing_keys):
            if re.match('model.vision_tower', key):
                incompatible_keys.missing_keys.remove(key)
