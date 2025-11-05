import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *
from functools import partial
from transformers import GPT2Tokenizer, GPT2Model


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self, d_model=None, bottleneck=None, dropout=0.0, init_option="lora", 
                 adapter_scalar="0.1", adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None

        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        self.scale = nn.Parameter(torch.ones(1)) if adapter_scalar == "learnable_scalar" else float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option
        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        output = up + residual if add_residual else up
        return output


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        return F.dropout(x, training=self.training)


class Base(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.clip = load_clip(name=config.clip_model, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        
        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()

        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.clip.dtype if self.clip.dtype is not None else torch.float16
        self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)

        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        self.additional_visual_params = self.add_visual_tunable_params()
        output_dim = self.clip.visual.output_dim
        self.img_logit = self.clip.logit_scale.exp()
        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        # init vision proxy(attr,obj)
        attr_tokenized = self.tokenizer(self.attributes, context_length=8).cuda()
        self.attr_tf = self.clip.encode_text(attr_tokenized)
        obj_tokenized = self.tokenizer(self.classes, context_length=8).cuda()
        self.obj_tf = self.clip.encode_text(obj_tokenized)
        self.attr_proxy = nn.Parameter(self.attr_tf).cuda()  # v^a
        self.obj_proxy = nn.Parameter(self.obj_tf).cuda()  # v^o

        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)
        self.linear = nn.Linear(output_dim * 2, output_dim)
        self.attr_ca = CrossResidualAttentionBlock(768, 16)
        self.obj_ca = CrossResidualAttentionBlock(768, 16)
        self.com_ca = CrossResidualAttentionBlock(768, 16)

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width, 
                                        bottleneck=self.config.adapter_dim, 
                                        dropout=self.config.adapter_dropout)
                                for _ in range(adapter_num)])
        return params

    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + 
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD
        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def construct_soft_prompt(self):
        token_ids = self.tokenizer(self.config.prompt_template, context_length=self.config.context_length).cuda()
        
        tokenized = torch.cat([self.tokenizer(tok, context_length=self.config.context_length) 
                              for tok in self.attributes + self.classes])
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros((len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)))

        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip.dtype)

        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]

        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(class_token_ids.cuda()).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)

        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1:len(self.comp_ctx_vectors) + 1, :] = self.comp_ctx_vectors.type(self.clip.dtype)

        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1:len(self.attr_ctx_vectors) + 1, :] = self.attr_ctx_vectors.type(self.clip.dtype)

        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1:len(self.obj_ctx_vectors) + 1, :] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor

    def forward(self, batch, idx, return_features=False):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        l, _ = idx.shape
        token_tensors = self.construct_token_tensors(idx)

        batch_img, patch_img = self.encode_image(batch_img.type(self.clip.dtype))
        cls_img, cls_patch_img = self.encode_image(batch_img.type(self.clip.dtype))

        text_features = self.encode_text(self.token_ids[0], token_tensors[0], enable_pos_emb=self.enable_pos_emb)[0]
        normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        attr_features = self.text_encoder(self.token_ids[1], token_tensors[1], enable_pos_emb=self.enable_pos_emb)[0]
        normalized_attr_text = attr_features / attr_features.norm(dim=-1, keepdim=True)

        obj_features = self.text_encoder(self.token_ids[2], token_tensors[2], enable_pos_emb=self.enable_pos_emb)[0]
        normalized_obj_text = obj_features / obj_features.norm(dim=-1, keepdim=True)

        # Image feature disentangling in Textual Prototype Learning
        attr_img, attr_weight = self.attr_ca(cls_img, attr_features)  # f_t^a, w^a
        obj_img, obj_weight = self.obj_ca(cls_img, obj_features)  # f_t^o, w^o
        logits_attr_weight = attr_weight * self.clip.logit_scale.exp()  # s^a
        logits_obj_weight = obj_weight * self.clip.logit_scale.exp()  # s^o
        batch_img_features = [batch_img, attr_img, obj_img]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        # Image feature disentangling in Visual Proxy Learning
        img_proxy_features = [cls_img, self.attr_disentangler(cls_img), self.obj_disentangler(cls_img)]  # f_v^c, f_v^a, f_v^o
        normalized_proxy_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in img_proxy_features]
        
        # Compositional vision proxy
        attr_idx, obj_idx = idx[:, 0], idx[:, 1]
        com_proxy = torch.zeros(l, 2, 768)
        com_proxy[:, 0, :] = self.attr_proxy[attr_idx].type(self.clip.dtype)
        com_proxy[:, 1, :] = self.obj_proxy[obj_idx].type(self.clip.dtype)
        combined_proxy = torch.cat((com_proxy[:, 0, :], com_proxy[:, 1, :]), dim=1).cuda()
        com_proxy = self.linear(combined_proxy)  # v^c
        
        normalized_com_proxy = com_proxy / com_proxy.norm(dim=-1, keepdim=True)
        normalized_attr_proxy = self.attr_proxy / self.attr_proxy.norm(dim=-1, keepdim=True)
        normalized_obj_proxy = self.obj_proxy / self.obj_proxy.norm(dim=-1, keepdim=True)

        logits = []

        # Calculate similarity between image features and vision proxies
        logits_com_proxy = (
                self.img_logit  # Scaling factor for cosine similarity
                * normalized_proxy_features[0]
                @ normalized_com_proxy.t()
            )
        logits_attr_proxy = (self.img_logit  # Scaling factor for cosine similarity
                * normalized_proxy_features[1]
                @ normalized_attr_proxy.t())
        logits_obj_proxy = (self.img_logit  # Scaling factor for cosine similarity
                * normalized_proxy_features[2]
                @ normalized_obj_proxy.t())
        
        # Calculate similarity between image features and textual prototypes
        logits_com_text = (self.clip.logit_scale.exp()  # Scaling factor for cosine similarity
                * normalized_img_features[0]
                @ normalized_text_features.t())
        logits_attr_text = (self.clip.logit_scale.exp()  # Scaling factor for cosine similarity
                * normalized_img_features[1]
                @ normalized_attr_text.t())
        
        logits_obj_text = (self.clip.logit_scale.exp()  # Scaling factor for cosine similarity
                * normalized_img_features[2]
                @ normalized_obj_text.t())
        logits_attr_text = logits_attr_text + logits_attr_weight #Attention Score-Based Probability for attribute
        logits_obj_text = logits_obj_text + logits_obj_weight #Attention Score-Based Probability for object

        logits.append(logits_com_proxy)
        logits.append(logits_attr_proxy)
        logits.append(logits_obj_proxy)
        logits.append(logits_com_text)
        logits.append(logits_attr_text)
        logits.append(logits_obj_text)
        
        if return_features:
            return logits, normalized_img_features[0], normalized_com_proxy, normalized_text_features
        else:
            return logits, normalized_attr_proxy, normalized_obj_proxy
