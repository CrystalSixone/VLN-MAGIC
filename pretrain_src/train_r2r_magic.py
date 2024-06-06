import os
import sys
import json
import argparse
import time
from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torch.cuda.amp as amp   # TODO

from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoModel

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config, postprocess_args

from data.loader import MetaLoader, PrefetchLoader, build_dataloader
from pretrain_src.data.dataset import R2RTextPathData, LoadZdict
from pretrain_src.data.dataset import read_img_features_from_h5py, read_img_features_from_h5py_multiprocess
from data.tasks import (
    MlmDataset, mlm_collate,
    MrcDataset, mrc_collate,
    SapDataset, sap_collate,
    CfpDataset, cfp_collate)

from model.pretrain_goat import GlocalTextPathCMTPreTraining

def create_dataloaders(
    data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts
):  
    dataloaders = {}
    for k, task_name in enumerate(data_cfg.tasks):
        if task_name == 'mlm':
            task_dataset = MlmDataset(nav_db, tok,)
            task_collate_fn = mlm_collate
        elif task_name == 'mrc':
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob, end_vp_pos_ratio=0.2)
            task_collate_fn = mrc_collate
        elif task_name == 'sap':
            task_dataset = SapDataset(nav_db, tok, end_vp_pos_ratio=0.2)
            task_collate_fn = sap_collate
        elif task_name == 'cfp':
            task_dataset = CfpDataset(nav_db, tok)
            task_collate_fn = cfp_collate
        else:
            raise ValueError(f'Undefined task {task_name}')

        LOGGER.info(f"{task_name}: {len(task_dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)
    return dataloaders


def main(opts):
    default_gpu, n_gpu, device, local_rank = set_cuda(opts)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}, local_rank: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1), opts.fp16, local_rank
            )
        )
 
    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    if default_gpu:
        save_training_meta(opts)
        TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Model config
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config['tasks'])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)
    model_config.cuda_first_device = opts.cuda_first_device

    data_cfg = EasyDict(opts.train_datasets['R2R'])
    if 'roberta' in model_config.lang_bert_name and data_cfg.name != 'RxR':
        data_cfg.train_traj_files = data_cfg.train_roberta_files
        data_cfg.val_seen_traj_files = data_cfg.val_seen_roberta_files
        data_cfg.val_unseen_traj_files = data_cfg.val_unseen_roberta_files
    LOGGER.info(f'   Use {model_config.lang_bert_name} tokenizer')
    
    # img feature
    model_config.img_file_type = 'hdf5'
    model_config.image_feat_size = 768
    data_cfg.img_ft_file = data_cfg.clip768_img_ft_file

    model_config.name = data_cfg.name
    LOGGER.info(f'   Use {model_config.img_feature_type} image features')

    # Preprocess the model configurations of student and teacher models.
    kdl_cfg = EasyDict(opts.kdl)
    student_model_config = copy.copy(model_config)
    if kdl_cfg.knowledge_distillation:
        teacher_model_config = model_config
        setattr(teacher_model_config, 'role', 'teacher')
        setattr(teacher_model_config, 'kd', kdl_cfg.knowledge_distillation)
        
        # Modify the dictionary
        teacher_model_config_attributes = vars(teacher_model_config)
        keys_to_modify = list(teacher_model_config_attributes.keys())
        # Iterate over the copied keys
        for attribute in keys_to_modify:
            value = teacher_model_config_attributes[attribute]
            if 'teacher' in attribute:
                # Modify the dictionary
                setattr(teacher_model_config, attribute[8:], value)
        setattr(teacher_model_config, "intermediate_size", int(teacher_model_config.hidden_size*teacher_model_config.mlp_ratio))
        setattr(teacher_model_config, "num_attention_heads", int(teacher_model_config.hidden_size/64))
            
    model_config_attributes = vars(model_config)
    student_model_config_attributes = vars(student_model_config)
    keys_to_modify = list(student_model_config_attributes.keys())
    # Iterate over the copied keys
    for attribute in keys_to_modify:
        value = student_model_config_attributes[attribute]
        if 'student' in attribute:
            # Modify the dictionary
            if kdl_cfg.knowledge_distillation:
                setattr(student_model_config, 'teacher_'+attribute[8:], model_config_attributes[attribute[8:]])
            setattr(student_model_config, attribute[8:], value)
    setattr(student_model_config, "intermediate_size", int(student_model_config.hidden_size*student_model_config.mlp_ratio))
    setattr(student_model_config, "num_attention_heads", int(student_model_config.hidden_size/64))
    setattr(student_model_config, 'role', 'student')
    setattr(student_model_config, 'kd', kdl_cfg.knowledge_distillation)
    student_model_config.kdl = kdl_cfg

    tokenizer = AutoTokenizer.from_pretrained(model_config.lang_bert_name)
    LOGGER.info(f'Tokenizer: {tokenizer}')
    
    # Prepare model
    # Loading teacher model's paramters
    if kdl_cfg.knowledge_distillation:
        if opts.teacher_checkpoint:
            teacher_checkpoint = torch.load(opts.teacher_checkpoint, map_location=lambda storage, loc: storage)
            LOGGER.info(f'   Use {opts.teacher_checkpoint } model to initialize the teacher model.')
        else:
            teacher_checkpoint = {}
            if opts.init_pretrained == 'bert':
                tmp = AutoModel.from_pretrained(model_config.lang_bert_name)
                for param_name, param in tmp.named_parameters():
                    teacher_checkpoint[param_name] = param
                if model_config.lang_bert_name == 'xlm-roberta-base':
                    # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
                    teacher_checkpoint['embeddings.token_type_embeddings.weight'] = torch.cat(
                        [teacher_checkpoint['embeddings.token_type_embeddings.weight']] * 2, 0
                    )
                del tmp
            elif opts.init_pretrained == 'meter':
                try:
                    tmp = torch.load('../datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
                except Exception:
                    tmp = torch.load('datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
                tmp = tmp['state_dict']
                for param_name, param in tmp.items():
                    if 'text_transformer.embeddings' in param_name:
                        param_name = param_name.replace('text_transformer.', 'bert.')
                        teacher_checkpoint[param_name] = param
                    elif 'text_transformer.encoder' in param_name:
                        param_name = param_name.replace('text_transformer.encoder', 'bert.lang_encoder')
                        if model_config.jump_init_txt:
                            param_name_list = param_name.split('.')
                            txt_layer_num = int(param_name_list[3])
                            if txt_layer_num % 2 == 0:
                                txt_layer_num = str(int(txt_layer_num/2))
                                param_name_list[3] = txt_layer_num
                                param_name = '.'.join(param_name_list)
                        teacher_checkpoint[param_name] = param
                    elif 'cross_modal_image_layers' in param_name:
                        param_name1 = param_name.replace('cross_modal_image_layers', 'bert.local_encoder.encoder.crossattention')
                        param_name2 = param_name.replace('cross_modal_image_layers', 'bert.global_encoder.encoder.crossattention')
                        teacher_checkpoint[param_name1] = teacher_checkpoint[param_name2] = param
                    else:
                        teacher_checkpoint[param_name] = param
                del tmp
            LOGGER.info(f'   Use {opts.init_pretrained } model to initialize the teacher model.')
        
    # Initialize the student model
    student_checkpoint = {}
    if opts.student_checkpoint:
        student_checkpoint = torch.load(opts.student_checkpoint, map_location=lambda storage, loc: storage)
    elif len(opts.student_checkpoint)==0:
        student_checkpoint = {}
        if opts.init_pretrained == 'bert':
            tmp = AutoModel.from_pretrained(model_config.lang_bert_name)
            for param_name, param in tmp.named_parameters():
                student_checkpoint[param_name] = param
            if model_config.lang_bert_name == 'xlm-roberta-base':
                # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
                student_checkpoint['embeddings.token_type_embeddings.weight'] = torch.cat(
                    [student_checkpoint['embeddings.token_type_embeddings.weight']] * 2, 0
                )
            del tmp
        elif opts.init_pretrained == 'meter':
            try:
                tmp = torch.load('../datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
            except Exception:
                tmp = torch.load('datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
            tmp = tmp['state_dict']
            for param_name, param in tmp.items():
                if 'text_transformer.embeddings' in param_name:
                    param_name = param_name.replace('text_transformer.', 'bert.')
                    student_checkpoint[param_name] = param
                elif 'text_transformer.encoder' in param_name:
                    param_name = param_name.replace('text_transformer.encoder', 'bert.lang_encoder')
                    if model_config.jump_init_txt:
                        param_name_list = param_name.split('.')
                        txt_layer_num = int(param_name_list[3])
                        if txt_layer_num % 2 == 0:
                            txt_layer_num = str(int(txt_layer_num/2))
                            param_name_list[3] = txt_layer_num
                            param_name = '.'.join(param_name_list)
                    student_checkpoint[param_name] = param
                elif 'cross_modal_image_layers' in param_name:
                    param_name1 = param_name.replace('cross_modal_image_layers', 'bert.local_encoder.encoder.crossattention')
                    param_name2 = param_name.replace('cross_modal_image_layers', 'bert.global_encoder.encoder.crossattention')
                    student_checkpoint[param_name1] = student_checkpoint[param_name2] = param
                else:
                    student_checkpoint[param_name] = param
            del tmp

    model_class = GlocalTextPathCMTPreTraining
    
    # Init models
    if kdl_cfg.knowledge_distillation:
        teacher_model = model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=teacher_model_config, state_dict=teacher_checkpoint
        )
        if kdl_cfg.train_teacher:
            teacher_model.train()
        else:
            teacher_model.eval()
        teacher_model = wrap_model(teacher_model, device, local_rank)
        del teacher_checkpoint
        LOGGER.info(f'KD! Teacher Trainable: {kdl_cfg.train_teacher}.')
    else:
        teacher_model = None
        
    LOGGER.info(f'kdl_adaptive_ability_weight: {kdl_cfg.kdl_adaptive_ability_weight}')

    student_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, config=student_model_config, state_dict=student_checkpoint
    )
    LOGGER.info(f'   Load student model successfully.')
    student_model.train()
    set_dropout(student_model, opts.dropout)
    student_model = wrap_model(student_model, device, local_rank)
    del student_checkpoint
    
    LOGGER.info(f'   Start loading image files ...')
    img_ft_db = read_img_features_from_h5py_multiprocess(data_cfg.img_ft_file, model_config.image_feat_size, num_processes=opts.n_process)

    aug_img_db = read_img_features_from_h5py_multiprocess(data_cfg.aug_img_file, model_config.image_feat_size, num_processes=opts.n_process)

    LOGGER.info(f'   Load image files done.')

    # Intervention
    z_dicts = None
    if model_config.do_back_img or model_config.do_back_txt:
        ZdictReader = LoadZdict(data_cfg.img_zdict_file,data_cfg.instr_zdict_file)
        z_dicts = defaultdict(lambda:None)
        if model_config.do_back_img:
            img_zdict = ZdictReader.load_img_tensor()
            z_dicts['img_zdict'] = img_zdict
        if model_config.do_back_txt:
            instr_zdict = ZdictReader.load_instr_tensor()
            z_dicts['instr_zdict'] = instr_zdict
    
    LOGGER.info(f'  Loading training datasets ...')
    # load data training set
    train_nav_db = R2RTextPathData(
        data_cfg.train_traj_files, img_ft_db,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, in_memory=True,
        cat_file=data_cfg.cat_file,
        args=model_config, tok=tokenizer,
        aug_img_db=aug_img_db,
        z_dicts=z_dicts,
        n_process=opts.n_process
    )
    LOGGER.info(f'  Loading training datasets done.')
    val_nav_db = R2RTextPathData(
        data_cfg.val_seen_traj_files, img_ft_db,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, in_memory=True,
        cat_file=data_cfg.cat_file,
        args=model_config, tok=tokenizer,
        aug_img_db=aug_img_db,
        z_dicts=z_dicts,
        n_process=opts.n_process
    )
    LOGGER.info(f'  Loading val_seen datasets done.')
    val2_nav_db = R2RTextPathData(
        data_cfg.val_unseen_traj_files, img_ft_db,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, in_memory=True,
        cat_file=data_cfg.cat_file,
        args=model_config, tok=tokenizer,
        aug_img_db=aug_img_db,
        z_dicts=z_dicts,
        n_process=opts.n_process
    )
    LOGGER.info(f'  Loading val_unseen datasets done.')

    # Build data loaders
    train_dataloaders = create_dataloaders(
        data_cfg, train_nav_db, tokenizer, True, device, opts
    )
    val_dataloaders = create_dataloaders(
        data_cfg, val_nav_db, tokenizer, False, device, opts
    )
    val2_dataloaders = create_dataloaders(
        data_cfg, val2_nav_db, tokenizer, False, device, opts
    )
    meta_loader = MetaLoader(
        train_dataloaders,
        accum_steps=opts.gradient_accumulation_steps,
        distributed=opts.local_rank != -1,
        device=device
    )
    meta_loader = PrefetchLoader(meta_loader, device)

    # Prepare optimizer
    optimizer = build_optimizer(student_model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}

    if opts.fp16:
        grad_scaler = amp.GradScaler()

    global_step = 0
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size if opts.local_rank == -1 else opts.train_batch_size * opts.world_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {}
    for task in train_dataloaders.keys():
        task2loss[task] = {
            kdl_task: RunningMeter(f'loss/{task}/{kdl_task}')
                for kdl_task in kdl_cfg.kdl_tasks
        }
        if kdl_cfg.knowledge_distillation:
            task2loss[task]['kdl_loss'] = RunningMeter(f'loss/{task}/supervised_loss')
                
        task2loss[task]['supervised_loss'] = RunningMeter(f'loss/{task}/supervised_loss')
        task2loss[task]['total_loss'] = RunningMeter(f'loss/{task}/total_loss')
    

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start_time = time.time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    max_unseen_facc = 0
    max_unseen_iter = 0
    
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f'------Step {global_step}: start validation seen------')
        validate(student_model, val_dataloaders, setname='_seen', tem=model_config.cfp_temperature)
        LOGGER.info(f'------Step {global_step}: start validation unseen------')
        validate(student_model, val2_dataloaders, setname='_unseen', tem=model_config.cfp_temperature)
        model_saver.save_latest(student_model, global_step)

def validate(model, val_dataloaders, setname='', max_metrix=None, tem=None):
    model.eval()
    max_update_flag = False
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate val{setname} on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader)
        elif task.startswith('sap'):
            val_log = validate_sap(model, loader)
            if setname == '_unseen' and max_metrix is not None:
                if val_log['facc'] >= max_metrix:
                    max_metrix = val_log['facc']
                    max_update_flag = True
        elif task.startswith('cfp'):
            val_log = validate_cfp(model, loader, tem)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'val{setname}_{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scalar_dict(
            {f'valid{setname}_{task}/{k}': v for k, v in val_log.items()}
        )
    model.train()
    if max_metrix is not None:
        return max_metrix, max_update_flag


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        outputs = model(batch, task='mlm', compute_loss=False)
        scores = outputs['predict']
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_word = sum(all_gather(n_word))
    tot_time = time.time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct

@torch.no_grad()
def validate_mrc(model, val_loader):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        view_logits, view_targets, _, _ = model(batch, task='mrc', compute_loss=False)
        view_logprobs = F.log_softmax(view_logits, dim=-1)
        loss = F.kl_div(view_logprobs, view_targets, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(view_logits, view_targets)
        val_loss += loss.item()
        n_feat += batch['vp_view_mrc_masks'].sum().item()
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log
    
@torch.no_grad()
def validate_sap(model, val_loader):
    LOGGER.info("start running SAP validation...")
    val_gloss, val_lloss, val_floss = 0, 0, 0
    n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        outputs = model(batch, task='sap', compute_loss=False)
        global_logits, local_logits, fused_logits, global_act_labels, local_act_labels = outputs['global_logits'], outputs['local_logits'], \
            outputs['fused_logits'], outputs['global_act_labels'], outputs['local_act_labels']
        val_gloss += F.cross_entropy(global_logits, global_act_labels, reduction='sum').data.item()
        val_lloss += F.cross_entropy(local_logits, local_act_labels, reduction='sum').data.item()
        val_floss += F.cross_entropy(fused_logits, global_act_labels, reduction='sum').data.item()
        n_gcorrect += torch.sum(torch.argmax(global_logits, 1) == global_act_labels).item()
        n_lcorrect += torch.sum(torch.argmax(local_logits, 1) == local_act_labels).item()
        n_fcorrect += torch.sum(torch.argmax(fused_logits, 1) == global_act_labels).item()
        n_data += len(global_act_labels)

    n_data = sum(all_gather(n_data))
    val_gloss = sum(all_gather(val_gloss)) / n_data
    val_lloss = sum(all_gather(val_lloss)) / n_data
    val_floss = sum(all_gather(val_floss)) / n_data
    gacc = sum(all_gather(n_gcorrect)) / n_data
    lacc = sum(all_gather(n_lcorrect)) / n_data
    facc = sum(all_gather(n_fcorrect)) / n_data
    
    tot_time = time.time()-st
    val_log = {'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
               'gacc': gacc, 'lacc': lacc, 'facc': facc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"gacc: {gacc*100:.2f}, lacc: {lacc*100:.2f}, facc: {facc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_cfp(model, val_loader, temperature):
    LOGGER.info("start running CFP validation...")
    val_gloss, val_lloss, val_floss = 0, 0, 0
    n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        gmap_outputs, vp_outputs, fused_outputs, txt_outputs = \
            model(batch, task='cfp', compute_loss=False)

        target_sim = torch.arange(len(gmap_outputs)).cuda()
        
        gmap_txt_sim = ( gmap_outputs @ txt_outputs.T ) / temperature
        global_txt_losses = (F.cross_entropy(gmap_txt_sim, target_sim, reduction='sum') +\
                                F.cross_entropy(gmap_txt_sim.T, target_sim.T, reduction='sum')) / 2.0

        vp_txt_sim = ( vp_outputs @ txt_outputs.T ) / temperature
        vp_txt_losses = (F.cross_entropy(vp_txt_sim, target_sim, reduction='sum') +\
                                F.cross_entropy(vp_txt_sim.T, target_sim.T, reduction='sum')) / 2.0
        
        fused_txt_sim = ( fused_outputs @ txt_outputs.T) / temperature
        fused_txt_losses = (F.cross_entropy(fused_txt_sim, target_sim, reduction='sum') +\
                                F.cross_entropy(fused_txt_sim.T, target_sim.T, reduction='sum')) / 2.0
        

        val_gloss += global_txt_losses.item()
        val_lloss += vp_txt_losses.item()
        val_floss += fused_txt_losses.item()
        n_gcorrect += torch.sum(torch.argmax(gmap_txt_sim, 1) == target_sim).item()
        n_lcorrect += torch.sum(torch.argmax(vp_txt_sim, 1) == target_sim).item()
        n_fcorrect += torch.sum(torch.argmax(fused_txt_sim, 1) == target_sim).item()
        n_data += len(target_sim)

        del target_sim

    n_data = sum(all_gather(n_data))
    val_gloss = sum(all_gather(val_gloss)) / n_data
    val_lloss = sum(all_gather(val_lloss)) / n_data
    val_floss = sum(all_gather(val_floss)) / n_data
    gacc = sum(all_gather(n_gcorrect)) / n_data
    lacc = sum(all_gather(n_lcorrect)) / n_data
    facc = sum(all_gather(n_fcorrect)) / n_data
    
    tot_time = time.time()-st
    val_log = {'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
               'gacc': gacc, 'lacc': lacc, 'facc': facc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"gacc: {gacc*100:.2f}, lacc: {lacc*100:.2f}, facc: {facc*100:.2f}")
    return val_log

def build_args():
    parser = load_parser()

    opts = parse_with_config(parser) # This is pretrain config
    postprocess_args(opts)

    return opts

if __name__ == '__main__':
    args = build_args()
    main(args)
