import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vlnbert_init import get_vlnbert_models

def convert_attn(input_attn, flat_shape=False):
    if isinstance(input_attn, tuple):
        input_attn = torch.cat(input_attn, dim=-1)
        if flat_shape:
            bs, head_nums, len_q, len_k = input_attn.size()
            input_attn = input_attn.view(bs * head_nums, len_q * len_k)
    return input_attn
    
class VLNBert(nn.Module):
    def __init__(self, args, role='student'):
        super().__init__()
        print(f'\nInitalizing the {role} model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None, role=role)
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)
        
        if mode == 'language':         
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            if not batch['already_dropout']:
                batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'reverie_obj_img_fts' in batch:
                batch['reverie_obj_img_fts'] = self.drop_env(batch['reverie_obj_img_fts'])
            pano_embeds, pano_masks, pano_fused_embeds, pano_attns = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks, pano_fused_embeds, pano_attns

        else:
            outs = self.vln_bert(mode, batch)
            return outs

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()