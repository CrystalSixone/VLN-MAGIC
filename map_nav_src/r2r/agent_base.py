import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.distributed import is_default_gpu
from utils.logger import print_progress
from utils.data import PickSpecificWords

from utils.kd_loss import kd_loss, dkd_loss, mse_loss, exponential_decay, invert_normalized_losses

class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, last_epoch=-1):
        self.warmup_iters = warmup_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [base_lr * (self.last_epoch + 1) / self.warmup_iters for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]

    
class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, z_dicts={}, z_front_dict={}, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
         
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(test=True,z_dicts=z_dicts,z_front_dict=z_front_dict,**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout(test=True,z_dicts=z_dicts,z_front_dict=z_front_dict,**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    env_actions = {
      'left': (0, -1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0, -1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0, tok=None):
        super().__init__(env)
        self.args = args
        self.feature_size = self.args.feature_size
        self.tok = tok

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # update back_txt_dict during training
        if args.z_instr_update:
            self.word_picker = PickSpecificWords(cat_file=args.cat_file)
            self.instr_specific_dict = defaultdict(lambda:[])

        # Models
        self._build_model()
        
        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        if self.args.train_kdl:
            self.teacher_models = (self.teacher_vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) 

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)
        
        if self.args.train_kdl and self.args.train_kdl_teacher:
            self.teacher_vln_bert_optimizer = optimizer(self.teacher_vln_bert.parameters(), lr=self.args.t_lr)

        if self.args.use_lr_sch:
            if self.args.use_warm_up:
                self.vln_bert_warmup_sch = WarmUpLR(self.vln_bert_optimizer, args.warmup_iters)
                self.vln_bert_lr_sch = CosineAnnealingLR(self.vln_bert_optimizer, T_max=(args.iters - args.warmup_iters), eta_min=args.lr*0.1)
            else:
                self.vln_bert_lr_sch = CosineAnnealingLR(self.vln_bert_optimizer, T_max=args.iters, eta_min=args.lr*0.1)

        # Evaluations
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='none')
        
        # Knowledge distillation
        if self.args.train_kdl:  
            if self.args.kdl_feat_loss == 'mse':
                self.kdl_feat_loss = mse_loss
            elif self.args.kdl_feat_loss == 'kl':
                self.kdl_feat_loss = kd_loss
            
            if self.args.kdl_attn_loss == 'mse':
                self.kdl_attn_loss = mse_loss
            elif self.args.kdl_attn_loss == 'kl':
                self.kdl_attn_loss = kd_loss
            
            if self.args.kdl_logit_loss == 'kd':
                self.kdl_logit_loss = kd_loss
            elif self.args.kdl_logit_loss == 'dkd':
                self.kdl_logit_loss = dkd_loss
                
            if self.args.teacher_sample_hard_mining:
                if self.args.t_sample_preprocess == 'exp':
                    self.t_sample_preprocess = exponential_decay
                elif self.args.t_sample_preprocess == 'norm':
                    self.t_sample_preprocess = invert_normalized_losses

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        raise NotImplementedError('child class should implement _build_model: self.vln_bert & self.critic')

    def test(self, use_dropout=False, feedback='argmax', iters=None, 
             z_dicts={}, z_front_dict=None, role='student', t_z_dicts={},
             t_z_front_dict=None,test_teacher=False, ensemble_n=1):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        use_dropout = use_dropout if ensemble_n == 1 else True
        if role == 'student':
            if use_dropout:
                self.vln_bert.train()
                if ensemble_n > 1:
                    # ensemble model test
                    for name, param in self.vln_bert.vln_bert.named_parameters():
                        param.requires_grad = False
                    self.rollout = self.ensemble_rollout
            else:
                self.vln_bert.eval()
        elif role == 'teacher':
            if use_dropout:
                self.teacher_vln_bert.train()
                if ensemble_n > 1:
                    # ensemble model test
                    for name, param in self.teacher_vln_bert.vln_bert.named_parameters():
                        param.requires_grad = False
                    self.rollout = self.ensemble_rollout
            else:
                self.teacher_vln_bert.eval()
        
        super().test(iters=iters, z_dicts=z_dicts, z_front_dict=z_front_dict,
                t_z_dicts=t_z_dicts, t_z_front_dict=t_z_front_dict,
                test_teacher=test_teacher, ensemble_n=ensemble_n)

    def train(self, n_iters, feedback='teacher', z_dicts={}, z_front_dict={}, t_z_dicts={}, 
              acc_grads=None, t_acc_grads=None, cur_iter=0, **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        if self.args.train_kdl:
            if self.args.train_kdl_teacher:
                self.teacher_vln_bert.train() # train the teacher
            else:
                self.teacher_vln_bert.eval() # freeze the teacher during training
        self.critic.train()
        
        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            if self.args.train_kdl_teacher:
                self.teacher_vln_bert_optimizer.zero_grad()

            self.loss = 0
            self.t_loss = 0
            if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=1., train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                )
            elif 'dagger' in self.args.train_alg:
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, t_z_dicts=t_z_dicts, **kwargs
                    ) 
                self.feedback = 'expl_sample' if self.args.expl_sample else 'sample' 
                self.rollout(train_ml=1, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, t_z_dicts=t_z_dicts, **kwargs)
            else:
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                    )
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs)

            if self.args.train_kdl_teacher:
                self.loss.backward(retain_graph=True)
            else:
                self.loss.backward()
            
            if self.args.kdl_adaptive_ability_weight and self.args.kdl_adaptive_ability_weight_type == 'grad':
                acc_grads = self.compute_multiSubject_grad(acc_grads, role='student')
            
            if self.args.train_kdl and self.args.train_kdl_teacher:
                self.t_loss.backward()
                if self.args.kdl_adaptive_ability_weight and self.args.kdl_adaptive_ability_weight_type == 'grad':
                    t_acc_grads = self.compute_multiSubject_grad(t_acc_grads, role='teacher')
                
                torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                self.vln_bert_optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.teacher_vln_bert.parameters(), 40.)
                self.teacher_vln_bert_optimizer.step()
            else:
                torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                self.vln_bert_optimizer.step()
            
            
            if self.args.use_lr_sch:
                if self.args.use_warm_up:
                    if cur_iter < self.args.warmup_iters:
                        self.vln_bert_warmup_sch.step()
                    else:
                        self.vln_bert_lr_sch.step(cur_iter-self.args.warmup_iters)
                else:
                    self.vln_bert_lr_sch.step(cur_iter)
                    
                self.logs['lr'].append(self.vln_bert_optimizer.param_groups[0]['lr'])

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)
        
        return acc_grads, t_acc_grads

    def save(self, epoch, path, role='student'):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
            }
        if role == 'student':
            all_tuple = [("vln_bert", self.vln_bert)]
        elif role == 'teacher':
            all_tuple = [("vln_bert", self.teacher_vln_bert)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path, role='student',train_kdl_teacher=False):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path, map_location=lambda storage, loc: storage)

        def recover_state(name, model, optimizer, role='student'):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if train_kdl_teacher and role == 'teacher':
                import copy
                new_state_dict = copy.copy(state_dict)
                for k, v in state_dict.items():
                    if k.split('.')[1] in ['txt_emb_w','vp_txt_w','gmap_txt_w','local_cross_w','global_cross_w','kdl_img_w','kdl_avg_img_w']:
                        del new_state_dict[k]
                state_dict = new_state_dict

            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                if list(model_keys)[0].startswith('module.') and (not list(load_keys)[0].startswith('module.')):
                    state_dict = {'module.'+k: v for k, v in state_dict.items()}
                same_state_dict = {}
                extra_keys = []
                for k, v in state_dict.items():
                    if k in model_keys:
                        same_state_dict[k] = v
                    else:
                        extra_keys.append(k)
                state_dict = same_state_dict
                print('Extra keys in state_dict: %s' % (', '.join(extra_keys)))
            state.update(state_dict)
            model.load_state_dict(state, strict=(not train_kdl_teacher)) # !!!
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        if role == 'student':
            all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer, role)]
        elif role == 'teacher':
            all_tuple = [("vln_bert", self.teacher_vln_bert, self.vln_bert_optimizer, role)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1


