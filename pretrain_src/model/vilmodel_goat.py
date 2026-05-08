import json
import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad

from .Bert_backbone import *

def convert_attn(input_attn, flat_shape=False):
    if isinstance(input_attn, tuple):
        input_attn = torch.cat(input_attn, dim=-1)
        if flat_shape:
            bs, head_nums, len_q, len_k = input_attn.size()
            input_attn = input_attn.view(bs * head_nums, len_q * len_k)
    return input_attn
    
class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks,
            output_attentions=True)
            txt_embeds, txt_attn = temp_output[0], temp_output[1]
            txt_attn = convert_attn(txt_attn)
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds, txt_attn

class LanguageEncoderDo(nn.Module):
    # add intervention
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)

        # BERT
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
 
        return txt_embeds


class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))

        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_fts))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens

class CausalImageEmbeddings(nn.Module):
    ''' Causal learning
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

        ''' For interventional image
        '''
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size+3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        if config.name != 'REVERIE' and config.name != 'SOON':
            self.img_self_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        self.do_back_img = config.do_back_img
        if self.do_back_img:
            self.do_img_before_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.do_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.do_img_attn = BertAttention(config)
            self.do_img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.do_img_concat_layernorm = BertLayerNorm(config.hidden_size, eps=1e-12)

            if self.config.do_imgobj_type == 'type_2':
                if self.config.do_add_method == 'door':
                    self.sigmoid = nn.Sigmoid()
                elif self.config.do_add_method == 'concat':
                    self.do_concat_img_linear = nn.Linear(config.hidden_size*2, config.hidden_size)

        '''For reverie'''
        if self.config.name == 'REVERIE' or self.config.name == 'SOON':
            self.obj_name_linear = nn.Embedding(config.obj_name_vocab_size, config.hidden_size)
            self.obj_reverie_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_reverie_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.nav_type_embedding = nn.Embedding(3, config.hidden_size)
            self.pano_encoder = create_transformer_encoder(
                        config, config.num_pano_layers, norm=True
                    )
        else:
            self.nav_type_embedding = nn.Embedding(2, config.hidden_size)

        '''For global map aggregation
        '''
        if config.adaptive_pano_fusion: 
            self.adaptive_pano_attn = nn.Linear(config.hidden_size,1) # 768 -> 1
            self.adaptive_pano_act = ACT2FN[config.hidden_act]
            self.adaptive_softmax = nn.Softmax(dim=1)

        # 0: objects, 1: navigable
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        '''for knowledge distillation'''
        self.role = config.role
        self.kd = config.kd
        if self.role == 'student' and self.kd:
            self.kdl_img_w = nn.Linear(config.hidden_size, config.teacher_hidden_size)
            self.kdl_avg_img_w = nn.Linear(config.hidden_size, config.teacher_hidden_size)
        
    def forward(
        self, traj_view_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, type_embed_layer, 
        traj_reverie_obj_fts=None, traj_reverie_obj_lens=None,
        traj_reverie_obj_locs=None, 
        traj_reverie_obj_names=None
    ):
        view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if self.config.name != 'REVERIE' and self.config.name != 'SOON':
            view_img_embeds = view_img_embeds + self.loc_layer_norm(self.loc_linear(traj_loc_fts))
        
        img_masks = gen_seq_masks(traj_vp_view_lens)
        extended_img_masks = extend_neg_masks(img_masks)

        if self.config.name != 'REVERIE' and self.config.name != 'SOON':
            view_img_embeds = self.dropout(view_img_embeds)
            view_img_embeds, view_img_attns = self.img_self_encoder(
                view_img_embeds, src_key_padding_mask=img_masks.logical_not()
            )

        '''For REVERIE'''
        if traj_reverie_obj_fts is not None:
            reverie_obj_img_embeds = self.obj_reverie_linear(traj_reverie_obj_fts)
            if self.config.use_obj_name:
                reverie_obj_img_embeds = reverie_obj_img_embeds + self.obj_name_linear(traj_reverie_obj_names)
            reverie_obj_img_embeds = self.obj_reverie_layer_norm(reverie_obj_img_embeds) 

            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                    view_img_embeds, reverie_obj_img_embeds, traj_vp_view_lens, traj_reverie_obj_lens
                ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            traj_vp_view_lens = traj_vp_view_lens + traj_reverie_obj_lens

            fused_img_reverie_embeds =  img_embeds +\
                    self.nav_type_embedding(traj_nav_types) +\
                    self.loc_layer_norm(self.loc_linear(traj_loc_fts))
            
            fused_img_reverie_embeds = self.layer_norm(fused_img_reverie_embeds)
            fused_img_reverie_embeds = self.dropout(fused_img_reverie_embeds)
                
            img_masks = gen_seq_masks(traj_vp_view_lens)
            view_img_embeds, view_img_attns = self.pano_encoder(
                fused_img_reverie_embeds, src_key_padding_mask=img_masks.logical_not()
            )
            view_img_attns = convert_attn(view_img_attns)
        
        # knowledge distillation
        if self.kd:
            kdl_fused_img_embeds = view_img_embeds.clone()
            if self.role == 'student' and self.kd:
                kdl_fused_img_embeds = self.kdl_img_w(kdl_fused_img_embeds)
        else:
            kdl_fused_img_embeds = None
        
        split_traj_embeds = torch.split(view_img_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_view_lens, traj_step_lens, 0)

        if self.config.adaptive_pano_fusion:
            traj_ori_embeds = view_img_embeds.clone()
            traj_fused_weight = self.adaptive_pano_attn(traj_ori_embeds) 
            traj_fused_weight_act = torch.tanh(traj_fused_weight) 
            traj_fused_weight_act = self.adaptive_softmax(traj_fused_weight_act)
            traj_fused_embeded_update = torch.mul(traj_ori_embeds,traj_fused_weight_act)
            traj_fused_embeds = torch.sum(traj_fused_embeded_update,dim=1)
            # knowledge distillation
            if self.kd:
                kdl_traj_fused_embeds = traj_fused_embeded_update.clone()
                if self.role == 'student' and self.kd:
                    kdl_traj_fused_embeds = self.kdl_avg_img_w(kdl_traj_fused_embeds)
            else:
                kdl_traj_fused_embeds = None

            split_traj_fused_embeds = torch.split(traj_fused_embeds, traj_step_lens, 0)
            return split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds,\
                kdl_fused_img_embeds, kdl_traj_fused_embeds, view_img_attns
        
        return split_traj_embeds, split_traj_vp_lens, None, kdl_fused_img_embeds, None, view_img_attns

class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)
        self.txt2img_encoder = CrossmodalEncoder(config)
        if 'cfp' in config.pretrain_tasks:
            self.tim_self_encoder = BertAttention(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds]) # x[-1]: the current observation
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        outputs = self.encoder(vp_embeds, vp_masks, txt_embeds, txt_masks)
        vp_embeds, vp_attns = outputs[0], outputs[1]
        vp_attns = convert_attn(vp_attns)
        
        return vp_embeds, vp_attns
    
    def forward_cfp(
        self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_masks = extend_neg_masks(vp_masks)
        outputs = self.tim_self_encoder(vp_embeds, vp_masks, output_attentions=True)
        vp_embeds, vp_attns = outputs[0], outputs[1]
        vp_attns = convert_attn(vp_attns)
        return vp_embeds, vp_attns

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        self.txt2img_encoder = CrossmodalEncoder(config)

        if 'cfp' in config.pretrain_tasks:
            self.tim_self_encoder = BertAttention(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        split_traj_fused_embeds=None
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                if split_traj_fused_embeds is not None:
                    visited_vp_fts[traj_vpids[i][t]] = split_traj_fused_embeds[i][t]
                else:
                    visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens,
        split_traj_fused_embeds=None
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            split_traj_fused_embeds=split_traj_fused_embeds
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None,
        split_traj_fused_embeds=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, split_traj_fused_embeds=split_traj_fused_embeds
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None
        
        outputs = self.encoder(
            gmap_embeds, gmap_masks, txt_embeds, txt_masks, 
            graph_sprels=graph_sprels
        )

        gmap_embeds, gmap_attns = outputs[0], outputs[1]
        gmap_attns = convert_attn(gmap_attns)

        return gmap_embeds, gmap_attns

    def forward_cfp(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None,
        split_traj_fused_embeds=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, split_traj_fused_embeds=split_traj_fused_embeds
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_masks = extend_neg_masks(gmap_masks)
        outputs = self.tim_self_encoder(
            gmap_embeds, gmap_masks,
            output_attentions=True
        )
        gmap_self_embeds, gmap_self_attns = outputs[0], outputs[1]
        gmap_self_attns = convert_attn(gmap_self_attns)
        return gmap_self_embeds, gmap_self_attns

class GlocalTextPathCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.role = config.role
        self.kd = config.kd
        self.embeddings = RobertaEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = CausalImageEmbeddings(config)
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        # knowledge distillation
        if self.role == 'student' and self.kd:
            self.txt_emb_w = nn.Linear(self.config.hidden_size, self.config.teacher_hidden_size)
            self.vp_txt_w = nn.Linear(self.config.hidden_size, self.config.teacher_hidden_size)
            self.gmap_txt_w = nn.Linear(self.config.hidden_size, self.config.teacher_hidden_size)
            self.fused_txt_w = nn.Linear(self.config.hidden_size, self.config.teacher_hidden_size)
            self.local_cross_w = nn.Linear(self.config.hidden_size, self.config.teacher_hidden_size)
            self.global_cross_w = nn.Linear(self.config.hidden_size, self.config.teacher_hidden_size)
            
            if config.kdl.kdl_adaptive_ability_weight:
                # let network to learn and adjust the weights for different losses
                self.kdl_txt_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.kdl_img_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.kdl_local_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.kdl_global_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.kdl_predict_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.init_weights()
            
    def forward(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        return_gmap_embeds=True,         
        z_img_features=None, z_img_pzs=None, traj_reverie_loc_fts=None, return_txt_embeds=False,
        traj_reverie_obj_names=None
    ):        
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_masks = gen_seq_masks(txt_lens)
        kdl_txt_embeds = None

        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)[0]
        if self.kd:
            kdl_txt_embeds = txt_embeds.clone()
            if self.role == 'student' and self.kd:
                kdl_txt_embeds = self.txt_emb_w(kdl_txt_embeds)
        txt_embeds, txt_attns = self.lang_encoder(txt_embeds, txt_masks)
        extended_txt_masks = extend_neg_masks(txt_masks)
        
        # trajectory embedding
        split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds,\
        kdl_fused_img_embeds, kdl_traj_fused_embeds, view_img_attns = self.img_embeddings(
            traj_view_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, self.embeddings.token_type_embeddings,
            traj_obj_img_fts, traj_vp_obj_lens, traj_reverie_loc_fts,
            traj_reverie_obj_names
        )
        
        # gmap embeds
        kdl_gmap_embeds, kdl_vp_embeds = None, None
        if return_gmap_embeds:
            gmap_embeds, gmap_attns = self.global_encoder(
                txt_embeds, txt_masks,
                split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
                gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=gmap_pair_dists,
                split_traj_fused_embeds=split_traj_fused_embeds,
            )
            kdl_gmap_embeds = gmap_embeds.clone()
            if self.role == 'student' and self.kd:
                kdl_gmap_embeds = self.global_cross_w(kdl_gmap_embeds)
        else:
            gmap_embeds, gmap_attns = None, None

        # vp embeds
        vp_embeds, vp_attns = self.local_encoder(
            txt_embeds, txt_masks,
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        if self.kd:
            kdl_vp_embeds = vp_embeds.clone()
            if self.role == 'student' and self.kd:
                kdl_vp_embeds = self.local_cross_w(kdl_vp_embeds)

        return gmap_embeds, vp_embeds, txt_embeds,\
            kdl_txt_embeds,\
            kdl_fused_img_embeds, kdl_traj_fused_embeds,\
            kdl_vp_embeds, kdl_gmap_embeds, \
            txt_attns, view_img_attns,\
            vp_attns, gmap_attns

    
    def forward_mlm(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        z_img_features=None, z_img_pzs=None,traj_reverie_loc_fts=None,traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None
        ):
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_masks = gen_seq_masks(txt_lens)

        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)[0]
        kdl_txt_embeds = txt_embeds.clone()
        if self.role == 'student' and self.kd:
            kdl_txt_embeds = self.txt_emb_w(kdl_txt_embeds)
        txt_embeds, txt_attns = self.lang_encoder(txt_embeds, txt_masks)
        extended_txt_masks = extend_neg_masks(txt_masks)
        
        split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds,\
        kdl_fused_img_embeds, kdl_traj_fused_embeds, view_img_attns = self.img_embeddings(
            traj_view_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, self.embeddings.token_type_embeddings,
            traj_obj_img_fts, traj_vp_obj_lens, traj_reverie_loc_fts,
            traj_reverie_obj_names
        )
        
        # gmap embeds
        gmap_input_embeds, gmap_masks = self.global_encoder.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, split_traj_fused_embeds=split_traj_fused_embeds
        )
        gmap_txt_embeds = txt_embeds
        extended_gmap_masks = extend_neg_masks(gmap_masks)

        gmap_txt_embeds = self.global_encoder.txt2img_encoder(
            gmap_txt_embeds, extended_txt_masks,
            gmap_input_embeds, extended_gmap_masks
        )[0] 
        kdl_gmap_txt_embeds = gmap_txt_embeds.clone()
        if self.role == 'student' and self.kd:
            kdl_gmap_txt_embeds = self.gmap_txt_w(kdl_gmap_txt_embeds)

        # vp embeds
        vp_input_embeds, vp_masks = self.local_encoder.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_txt_embeds = txt_embeds
        extended_vp_masks = extend_neg_masks(vp_masks)
        vp_txt_embeds = self.local_encoder.txt2img_encoder(
            vp_txt_embeds, extended_txt_masks, 
            vp_input_embeds, extended_vp_masks,
        )[0] 
        kdl_vp_txt_embeds = vp_txt_embeds.clone()
        if self.role == 'student' and self.kd:
            kdl_vp_txt_embeds = self.vp_txt_w(kdl_vp_txt_embeds)

        txt_embeds = gmap_txt_embeds + vp_txt_embeds
        kdl_fused_txt_embeds = txt_embeds.clone()
        if self.role == 'student' and self.kd:
            kdl_fused_txt_embeds = self.fused_txt_w(kdl_fused_txt_embeds)
        return txt_embeds, kdl_txt_embeds, kdl_vp_txt_embeds, kdl_gmap_txt_embeds, kdl_fused_txt_embeds,\
            kdl_fused_img_embeds, kdl_traj_fused_embeds, txt_attns, view_img_attns

    def forward_cfp(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        return_gmap_embeds=True,         
        traj_reverie_loc_fts=None, return_txt_embeds=False,
        traj_reverie_obj_names=None
    ):        
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_masks = gen_seq_masks(txt_lens)

        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)[0]
        kdl_txt_embeds = txt_embeds.clone()
        if self.role == 'student' and self.kd:
            kdl_txt_embeds = self.txt_emb_w(kdl_txt_embeds)
        txt_embeds, txt_attns = self.lang_encoder(txt_embeds, txt_masks)
        extended_txt_masks = extend_neg_masks(txt_masks)
        
        # trajectory embedding
        split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds,\
        kdl_fused_img_embeds, kdl_traj_fused_embeds, view_img_attns = self.img_embeddings(
            traj_view_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, self.embeddings.token_type_embeddings,
            traj_obj_img_fts, traj_vp_obj_lens, traj_reverie_loc_fts,
            traj_reverie_obj_names
        )
        
        # gmap embeds
        if return_gmap_embeds: 
            gmap_embeds, gmap_attns = self.global_encoder.forward_cfp(
                split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
                gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=gmap_pair_dists,
                split_traj_fused_embeds=split_traj_fused_embeds
            )
            kdl_gmap_embeds = gmap_embeds.clone()
            if self.role == 'student' and self.kd:
                kdl_gmap_embeds = self.global_cross_w(kdl_gmap_embeds)
        else:
            gmap_embeds, gmap_attns = None, None

        # vp embeds
        vp_embeds, vp_attns = self.local_encoder.forward_cfp(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        kdl_vp_embeds = vp_embeds.clone()
        if self.role == 'student' and self.kd:
            kdl_vp_embeds = self.local_cross_w(kdl_vp_embeds)

        return gmap_embeds, vp_embeds, txt_embeds,\
            kdl_txt_embeds,\
            kdl_fused_img_embeds, kdl_traj_fused_embeds,\
            kdl_vp_embeds, kdl_gmap_embeds,\
            txt_attns, view_img_attns,\
            vp_attns, gmap_attns
    
