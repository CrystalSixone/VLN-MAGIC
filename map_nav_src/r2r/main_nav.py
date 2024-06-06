import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src')
sys.path.append(root_path)
sys.path.append(current_path)
import json
import time
import numpy as np
import copy
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB, Tokenizer, KMeansPicker
from r2r.transpeaker import Speaker
from r2r.data_utils import construct_instrs
from r2r.env import R2RNavBatch
from r2r.parser import parse_args

from r2r.agent import GMapNavAgent
from data_utils import LoadZdict


def build_dataset(args, rank=0):
    # Load vocab for speaker
    try:
        train_vocab_file = os.path.join(root_path,'r2r','train_vocab.txt')
        with open(train_vocab_file) as f:
            vocab = [word.strip() for word in f.readlines()]
    except Exception:
        train_vocab_file = os.path.join(current_path,'r2r','train_vocab.txt') # for Debug
        with open(train_vocab_file) as f:
            vocab = [word.strip() for word in f.readlines()]
    speaker_tok = Tokenizer(vocab=vocab, encoding_length=args.max_instr_len)
    
    try:
        bert_tok = AutoTokenizer.from_pretrained('../datasets/pretrained/roberta')        
    except Exception:
        bert_tok = AutoTokenizer.from_pretrained(os.path.join(root_path,'datasets','pretrained','roberta')) # for Debug
    
    # For do-intervention
    if args.dataset == 'rxr':
        instr_z_file = args.rxr_instr_zdict_roberta_file
    else:
        instr_z_file = args.instr_zdict_file

    instr_zdict_file = args.backdoor_dict_file if len(args.backdoor_dict_file)>1 else instr_z_file
    ZdictReader = LoadZdict(args.img_zdict_file, instr_zdict_file) # teacher's instr_zdict. student's will be updated at first before training.
    z_dicts = defaultdict(lambda:None)
    if args.do_back_img:
        img_zdict = ZdictReader.load_img_tensor()
        z_dicts['img_zdict'] = img_zdict
    if args.do_back_txt:
        instr_zdict = ZdictReader.load_instr_tensor()
        z_dicts['instr_zdict'] = instr_zdict

    front_feat_loader = None
    t_front_feat_loader = None
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        front_feat_file = args.rxr_front_feat_file if args.dataset=='rxr' else args.front_feat_file
        front_feat_file = args.s_frontdoor_dict_file if len(args.s_frontdoor_dict_file) > 1 else front_feat_file
        
        front_feat_loader = KMeansPicker(front_feat_file,n_clusters=args.front_n_clusters)
        if args.train_kdl:
            t_front_feat_file = args.rxr_t_front_feat_file if args.dataset=='rxr' else args.t_front_feat_file
            t_front_feat_file = args.t_frontdoor_dict_file if len(args.t_frontdoor_dict_file) > 1 else front_feat_file
            t_front_feat_loader = KMeansPicker(t_front_feat_file,n_clusters=args.front_n_clusters)

    # Load features
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    
    # Use augmented features
    if args.use_aug_env:
        train_feat_db = [feat_db]
        if args.env_edit:
            print('use env_edit features!!')
            envedit_feat_db = ImageFeaturesDB(args.aug_img_ft_file_envedit, args.image_feat_size)
            train_feat_db.append(envedit_feat_db)    
        if len(train_feat_db) == 1:
            print('not assign augmented features!!!')
            train_feat_db = feat_db
    else:
        train_feat_db = feat_db

    dataset_class = R2RNavBatch

    if args.aug is not None: # trajectory & instruction aug
        aug_feat_db = train_feat_db
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len, for_debug=args.for_debug,
            tok=bert_tok, is_rxr=(args.dataset=='rxr')
        )
        aug_env = dataset_class(
            aug_feat_db, aug_instr_data, args.connectivity_dir, 
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            args=args, scanvp_cands_file=args.scanvp_cands_file
        )
    else:
        aug_env = None

    # Load the training dataset
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len, for_debug=args.for_debug,
        tok=bert_tok, is_rxr=(args.dataset=='rxr')
    )
    train_env = dataset_class(
        train_feat_db, train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
        args=args, scanvp_cands_file=args.scanvp_cands_file
    )

    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    if args.dataset == 'rxr':
        val_env_names.remove('val_train_seen')
        if not args.submit:
            val_env_names.remove('val_seen')
    
    if args.train_kdl_teacher:
        if args.for_debug:
            val_env_names = ['val_train_seen']
        else:
            if args.dataset != 'rxr':
                val_env_names.remove('val_train_seen')
    else:
        if args.for_debug:
            val_env_names = ['val_train_seen']
    
    if args.submit and args.dataset != 'rxr':
        val_env_names.append('test')
    
    if args.sim_env == 'tjark':
        val_env_names = ['train', 'test']
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,for_debug=args.for_debug,
            tok=bert_tok, is_rxr=(args.dataset=='rxr')
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            args=args, scanvp_cands_file=args.scanvp_cands_file
        )  
        val_envs[split] = val_env

    return train_env, val_envs, aug_env, bert_tok, speaker_tok, z_dicts, train_instr_data, front_feat_loader, t_front_feat_loader


def train(args, train_env, val_envs, aug_env=None, rank=-1, bert_tok=None, speaker_tok=None, z_dicts=None,train_instr_data=None,front_feat_loader=None,\
t_front_feat_loader=None,t_z_dicts=None):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank, tok=bert_tok)

    # resume file
    start_iter = 0
    if args.student_resume_file is not None:
        start_iter = listner.load(os.path.join(args.student_resume_file), role='student')
        if default_gpu:
            write_to_record_file(
                "\nLOAD the student model from {}, iteration {}".format(args.student_resume_file, start_iter),
                record_file
            )
        if args.train_kdl_teacher or args.sim_env == 'tjark':
            start_iter = 0
    
    if args.train_kdl and args.teacher_resume_file is not None:
        t_start_iter = listner.load(os.path.join(args.teacher_resume_file), role='teacher', train_kdl_teacher=args.train_kdl_teacher)
        if default_gpu:
            write_to_record_file(
                "\nLOAD the teacher model from {}, iteration ".format(args.teacher_resume_file, t_start_iter),
                record_file
            )
    
    # z_dicts
    z_front_dict = None
    t_z_front_dict = None
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        if len(args.frontdoor_dict_file) > 0:
            z_front_dict = front_feat_loader.read_tim_tsv(args.frontdoor_dict_file, return_dict=True)
        else:
            z_front_dict = front_feat_loader.random_pick_front_features(args, iter=0, save_file=False)

        if args.train_kdl:
            if len(args.t_frontdoor_dict_file) > 0:
                t_z_front_dict = t_front_feat_loader.read_tim_tsv(args.t_frontdoor_dict_file, return_dict=True)
            else:
                t_z_front_dict = t_front_feat_loader.random_pick_front_features(args, iter=0, save_file=False)

    
    if t_z_dicts is None:
        t_z_dicts = copy.deepcopy(z_dicts) # keep consistent

    if args.do_back_txt:
        if args.z_instr_update:
            z_dicts, landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict = listner.update_z_dict(train_instr_data,iter=0, z_dict=z_dicts, add_back_feat=False)
            if args.train_kdl_teacher:
                t_z_dicts, t_landmark_dict, t_direction_dict, t_landmark_pz_dict, t_direction_pz_dict = listner.update_z_dict(train_instr_data,iter=0, z_dict=t_z_dicts, role='teacher', add_back_feat=False)
        else:
            # for debug
            for key, value in z_dicts['instr_zdict'].items():
                z_dicts['instr_zdict'][key] = value[...,:args.student_hidden_size] 

    # first evaluation
    if args.eval_first:
        # 1. valid the student model
        loss_str = "validation the student model before training"
        for env_name, env in val_envs.items():
            start_time = time.time()
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None, role='student', z_dicts=z_dicts, z_front_dict=z_front_dict)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
            write_to_record_file("Finish valid %s in %s." %(env_name, timeSince(start_time, float(1)/args.iters)), record_file)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
            
            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "student_detail_%s.json" % (env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
        
        # 2. valid the teacher model
        if args.train_kdl and args.teacher_resume_file is not None:
            loss_str = "validation the teacher model before training"
            for env_name, env in val_envs.items():
                start_time = time.time()
                listner.env = env
                # Get validation distance from goal under test evaluation conditions
                listner.test(use_dropout=False, feedback='argmax', iters=None, role='teacher', t_z_dicts=t_z_dicts, t_z_front_dict=t_z_front_dict,
                test_teacher=True)
                preds = listner.get_results()
                # gather distributed results
                preds = merge_dist_results(all_gather(preds))
                if default_gpu:
                    score_summary, _ = env.eval_metrics(preds)
                    loss_str += ", %s " % env_name
                    for metric, val in score_summary.items():
                        loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file("Finish valid %s in %s." %(env_name, timeSince(start_time, float(1)/args.iters)), record_file)
            if default_gpu:
                write_to_record_file(loss_str, record_file)
                
                if args.submit:
                    json.dump(
                        preds,
                        open(os.path.join(args.pred_dir, "teacher_detail_%s.json" % (env_name)), 'w'),
                        sort_keys=True, indent=4, separators=(',', ': ')
                    )
        
        # return 

    if args.use_transpeaker:
        speaker = Speaker(args,train_env,speaker_tok)
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)
        print("Load speaker model successully.")
    else:
        speaker = None

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", "both": 0.}} 
    t_best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", "both": 0.}}
    if args.sim_env == 'tjark':
        best_val = {'test': {"spl": 0., "sr": 0., "state":"", "both": 0.}} 
        t_best_val = {'test': {"spl": 0., "sr": 0., "state":"", "both": 0.}}
    if args.dataset == 'rxr':
        best_val = {'val_unseen': {"nDTW": 0., "SDTW": 0., "state":"", "both": 0.}}
        t_best_val = {'val_unseen': {"nDTW": 0., "SDTW": 0., "state":"", "both": 0.}}
    
    if args.kdl_adaptive_ability_weight and args.kdl_adaptive_ability_weight_type == 'grad':
        acc_grads = defaultdict(lambda:0)
        t_acc_grads = defaultdict(lambda:0)
        current_iter_grads = {
            'txt': 1, 'img': 1, 'local': 1, 'global': 1, 'action': 1
        }
        t_current_iter_grads = {
            'txt': 1, 'img': 1, 'local': 1, 'global': 1, 'action': 1
        }
    else:
        acc_grads = current_iter_grads = None
        t_acc_grads = t_current_iter_grads = None
        
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        is_update = False

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            acc_grads, t_acc_grads = listner.train(interval, feedback=args.feedback, z_dicts=z_dicts, t_z_dicts=t_z_dicts, z_front_dict=z_front_dict, t_z_front_dict=t_z_front_dict, 
            acc_grads=acc_grads, t_acc_grads=t_acc_grads,
            current_iter_grads=current_iter_grads,t_current_iter_grads=t_current_iter_grads, cur_iter=iter)  # Train interval iters
        else:
            if args.accumulate_grad: # accumulateGrad
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    
                    listner.zero_grad()
                    listner.env = train_env
                    if speaker is not None:
                        speaker.env = train_env

                    # Train with GT data
                    listner.accumulate_gradient(args.feedback, z_dicts=z_dicts, t_z_dicts=t_z_dicts, z_front_dict=z_front_dict, t_z_front_dict=t_z_front_dict, current_iter_grads=current_iter_grads,t_current_iter_grads=t_current_iter_grads)
                    listner.env = aug_env
                    if speaker is not None:
                        speaker.env = aug_env

                    # Train with Back Translation
                    listner.accumulate_gradient(args.feedback, speaker=speaker, z_dicts=z_dicts, t_z_dicts=t_z_dicts, z_front_dict=z_front_dict, t_z_front_dict=t_z_front_dict, current_iter_grads=current_iter_grads,t_current_iter_grads=t_current_iter_grads)
                    acc_grads, t_acc_grads = listner.optim_step(acc_grads, t_acc_grads, cur_iter=iter)

                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)
            else:
                jdx_length = len(range(interval // (args.aug_times+1)))
                for jdx in range(interval // (args.aug_times+1)):
                    # Train with GT data
                    listner.env = train_env
                    listner.train(1, feedback=args.feedback,z_dicts=z_dicts, z_front_dict=z_front_dict, t_z_dicts=t_z_dicts,t_z_front_dict=t_z_front_dict, current_iter_grads=current_iter_grads,t_current_iter_grads=t_current_iter_grads)

                    # Train with Augmented data
                    listner.env = aug_env
                    listner.train(args.aug_times, feedback=args.feedback, speaker=speaker, z_dicts=z_dicts, t_z_dicts=t_z_dicts,z_front_dict=z_front_dict, t_z_front_dict=t_z_front_dict, current_iter_grads=current_iter_grads,t_current_iter_grads=t_current_iter_grads)

                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            if args.train_kdl:
                KDL_loss = sum(listner.logs['kdl_loss']) / max(len(listner.logs['kdl_loss']), 1)
                ML_loss = sum(listner.logs['ml_loss']) / max(len(listner.logs['ml_loss']), 1)
                writer.add_scalar("loss/KDL_loss", KDL_loss, idx)
                writer.add_scalar("loss/ML_loss", ML_loss, idx)
                
                if args.kdl_adaptive_ability_weight:
                    if args.kdl_adaptive_ability_weight_type == 'learned_weight':
                        # Calculate the average weights from the logged values
                        kdl_txt_weight_avg = sum(listner.logs['kdl_txt_weight']) / max(len(listner.logs['kdl_txt_weight']), 1)
                        kdl_img_weight_avg = sum(listner.logs['kdl_img_weight']) / max(len(listner.logs['kdl_img_weight']), 1)
                        kdl_local_weight_avg = sum(listner.logs['kdl_local_weight']) / max(len(listner.logs['kdl_local_weight']), 1)
                        kdl_global_weight_avg = sum(listner.logs['kdl_global_weight']) / max(len(listner.logs['kdl_global_weight']), 1)
                        kdl_predict_weight_avg = sum(listner.logs['kdl_predict_weight']) / max(len(listner.logs['kdl_predict_weight']), 1)

                        # Log the average weights
                        writer.add_scalar("kdl_adaptive_weight/kdl_txt_weight", kdl_txt_weight_avg, idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_img_weight", kdl_img_weight_avg, idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_local_weight", kdl_local_weight_avg, idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_global_weight", kdl_global_weight_avg, idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_predict_weight", kdl_predict_weight_avg, idx)

                    elif args.kdl_adaptive_ability_weight_type == 'grad':
                        writer.add_scalar("kdl_adaptive_weight/kdl_txt_weight", current_iter_grads['txt'], idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_img_weight", current_iter_grads['img'], idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_local_weight", current_iter_grads['local'], idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_global_weight", current_iter_grads['global'], idx)
                        writer.add_scalar("kdl_adaptive_weight/kdl_action_weight", current_iter_grads['action'], idx)
                        
                if args.train_kdl_teacher:
                    t_KDL_loss = sum(listner.logs['t_kdl_loss']) / max(len(listner.logs['kdl_loss']), 1)
                    t_ML_loss = sum(listner.logs['t_ml_loss']) / max(len(listner.logs['ml_loss']), 1)
                    t_IL_loss = sum(listner.logs['t_IL_loss']) / max(len(listner.logs['IL_loss']), 1)
                    writer.add_scalar("t_loss/KDL_loss", t_KDL_loss, idx)
                    writer.add_scalar("t_loss/ML_loss", t_ML_loss, idx)
                    writer.add_scalar("t_loss/IL_loss", t_IL_loss, idx)
                    
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            if args.use_lr_sch:
                LR = sum(listner.logs['lr']) / max(len(listner.logs['lr']), 1)
                writer.add_scalar("lr", LR,idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        t_loss_str = "iter {}".format(iter)

        if args.empty_cache:
            torch.cuda.empty_cache()
                        
        if args.z_instr_update and iter%(args.update_iter)==0:
            is_update = True
            if args.do_back_txt:
                z_dicts, landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict = listner.update_z_dict(train_instr_data,iter,z_dicts,front_dict=None,save_file=False,add_back_feat=True)
            if args.do_front_img or args.do_front_his or args.do_front_txt:
                z_front_dict = front_feat_loader.random_pick_front_features(args, iter, save_file=False)
        
        if args.train_kdl and args.kdl_adaptive_ability_weight and args.kdl_adaptive_ability_weight_type == 'grad' and iter%(args.aw_update_iter)==0:
            for k,v in acc_grads.items():
                acc_grads[k] /= args.update_iter
            current_iter_grads = copy.copy(acc_grads)
            acc_grads = defaultdict(lambda:0) # init
            if args.train_kdl_teacher:
                for k,v in acc_grads.items():
                    t_acc_grads[k] /= args.update_iter
                t_current_iter_grads = copy.copy(t_acc_grads)
                t_acc_grads = defaultdict(lambda:0)

        for env_name, env in val_envs.items():
            listner.env = env
            start_time = time.time()
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None, z_dicts=z_dicts, z_front_dict=z_front_dict, role='student')
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by spl
                if env_name in best_val:
                    if args.dataset == 'rxr':
                        target_metric = 'nDTW'
                        target_metric_2 = 'SDTW'
                    else:
                        target_metric = 'spl'
                        target_metric_2 = 'sr'

                    if score_summary[target_metric] + score_summary[target_metric_2] >= best_val[env_name]['both']:
                        best_val[env_name][target_metric] = score_summary[target_metric]
                        best_val[env_name][target_metric_2] = score_summary[target_metric_2]
                        best_val[env_name]['both'] = score_summary[target_metric] + score_summary[target_metric_2]
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s.pt" % (env_name)))

                        if args.z_instr_update and (not is_update):
                            if args.do_back_txt:
                                listner.save_backdoor_z_dict(landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict)
                                z_dicts, landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict = listner.update_z_dict(train_instr_data,iter, z_dicts, front_dict=None,save_file=False,add_back_feat=True)
                            if args.do_front_img or args.do_front_his or args.do_front_txt:
                                front_feat_loader.save_features(args, z_front_dict)
                                z_front_dict = front_feat_loader.random_pick_front_features(args, iter, save_file=False)
            
            write_to_record_file("Finish valid %s in %s." %(env_name, timeSince(start_time, float(iter)/args.iters)), record_file)
            # test the teacher model
            if args.train_kdl_teacher:
                start_time = time.time()
                listner.env = env
                listner.test(use_dropout=False, feedback='argmax', iters=None, t_z_dicts=t_z_dicts, t_z_front_dict=t_z_front_dict, role='teacher', test_teacher=True)
                preds = listner.get_results()
                preds = merge_dist_results(all_gather(preds))

                if default_gpu:
                    t_score_summary, _ = env.eval_metrics(preds)
                    t_loss_str += ", %s " % env_name
                    for metric, val in t_score_summary.items():
                        t_loss_str += ', %s: %.2f' % (metric, val)
                        writer.add_scalar('t_%s/%s' % (metric, env_name), score_summary[metric], idx)

                    # select model by spl
                    if env_name in t_best_val:
                        if args.dataset == 'rxr':
                            target_metric = 'nDTW'
                            target_metric_2 = 'SDTW'
                        else:
                            target_metric = 'spl'
                            target_metric_2 = 'sr'

                        if t_score_summary[target_metric] + t_score_summary[target_metric_2] >= t_best_val[env_name]['both']:
                            t_best_val[env_name][target_metric] = t_score_summary[target_metric]
                            t_best_val[env_name][target_metric_2] = t_score_summary[target_metric_2]
                            t_best_val[env_name]['both'] = t_score_summary[target_metric] + t_score_summary[target_metric_2]
                            t_best_val[env_name]['state'] = 'Iter %d %s' % (iter, t_loss_str)
                            listner.save(idx, os.path.join(args.ckpt_dir, "teacher_best_%s.pt" % (env_name)), role='teacher')

                            if args.z_instr_update and (not is_update):
                                if args.do_back_txt:
                                    listner.save_backdoor_z_dict(t_landmark_dict, t_direction_dict, t_landmark_pz_dict, t_direction_pz_dict, role='teacher')
                                    t_z_dicts, t_landmark_dict, t_direction_dict, t_landmark_pz_dict, t_direction_pz_dict = listner.update_z_dict(train_instr_data, iter, t_z_dicts, front_dict=None, save_file=False, role='teacher',add_back_feat=True)
                                if args.do_front_img or args.do_front_his or args.do_front_txt:
                                    t_front_feat_loader.save_features(args, t_z_front_dict, role='teacher')
                                    t_z_front_dict = t_front_feat_loader.random_pick_front_features(args, iter, save_file=False)

                write_to_record_file("Finish valid %s in %s." %(env_name, timeSince(start_time, float(iter)/args.iters)), record_file)
                
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict.pt"), role='student')
            if args.train_kdl_teacher:
                listner.save(idx, os.path.join(args.ckpt_dir, "teacher_latest_dict.pt"), role='teacher')

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("Student's BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)

            if args.train_kdl_teacher:
                write_to_record_file(
                    ('\n%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, t_loss_str)),
                    record_file
                )
                write_to_record_file("Teacher's BEST RESULT TILL NOW", record_file)
                for env_name in t_best_val:
                    write_to_record_file(env_name + ' | ' + t_best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1, z_dicts=None, front_feat_loader=None, t_front_feat_loader=None):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)
    
    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    
    '''1. valid the student model'''
    if args.student_resume_file is not None:
        print("Loaded the student model at iter %d from %s" % (
            agent.load(args.student_resume_file, role='student'), args.student_resume_file))
    
    # Load student's backdoor dictionary
    s_ZdictReader = LoadZdict(args.img_zdict_file, args.s_backdoor_dict_file)
    s_z_dicts = defaultdict(lambda:None)
    s_instr_zdict = s_ZdictReader.load_instr_tensor()
    s_z_dicts['instr_zdict'] = s_instr_zdict
    s_z_dicts['img_zdict'] = z_dicts['img_zdict']
    
    # Load front-door dictionary
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        if len(args.s_frontdoor_dict_file) > 0:
            z_front_dict =  front_feat_loader.read_tim_tsv(args.s_frontdoor_dict_file, return_dict=True)
        else:
            z_front_dict = front_feat_loader.random_pick_front_features(args, iter=0, save_file=False)
    else:
        z_front_dict = None

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        # if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
        #     continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters,z_dicts=s_z_dicts, z_front_dict=z_front_dict, role='student', ensemble_n=args.ensemble_n)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "student_%s_%s.json" % (prefix, env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    
    '''2. valid the teacher model'''
    if args.train_kdl and args.teacher_resume_file is not None:
        print("Loaded the teacher model at iter %d from %s" % (
            agent.load(args.teacher_resume_file, role='teacher'), args.teacher_resume_file))
    
        # Load front-door dictionary
        if args.do_front_img or args.do_front_his or args.do_front_txt:
            if len(args.t_frontdoor_dict_file) > 0:
                t_z_front_dict = t_front_feat_loader.read_tim_tsv(args.t_frontdoor_dict_file, return_dict=True)
            else:
                t_z_front_dict = t_front_feat_loader.random_pick_front_features(args, iter=0, save_file=False)
                t_front_feat_loader.save_features(args, t_z_front_dict, role='teacher')
        else:
            t_z_front_dict = None

        for env_name, env in val_envs.items():
            prefix = 'submit' if args.detailed_output is False else 'detail'
            # if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            #     continue
            agent.logs = defaultdict(list)
            agent.env = env

            iters = None
            start_time = time.time()
            agent.test(
                use_dropout=False, feedback='argmax', iters=iters,t_z_dicts=z_dicts, t_z_front_dict=t_z_front_dict, test_teacher=True, role='teacher', ensemble_n=args.ensemble_n)
            print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
            preds = agent.get_results(detailed_output=args.detailed_output)
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                if 'test' not in env_name:
                    score_summary, _ = env.eval_metrics(preds)
                    loss_str = "Env name: %s" % env_name
                    for metric, val in score_summary.items():
                        loss_str += ', %s: %.2f' % (metric, val)
                    write_to_record_file(loss_str+'\n', record_file)

                if args.submit:
                    json.dump(
                        preds,
                        open(os.path.join(args.pred_dir, "teacher_%s_%s.json" % (prefix, env_name)), 'w'),
                        sort_keys=True, indent=4, separators=(',', ': ')
                    )

def extract_cfp_features(args, train_env, z_dict, rank=0, bert_tok=None, save_file=True):
    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank, tok=bert_tok)

    if args.student_resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.student_resume_file), args.student_resume_file))
    
    agent.extract_cfp_features(train_env.data, z_dict=z_dict, save_file=save_file)

def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env, bert_tok, speaker_tok, z_dicts, train_instr_data, front_feat_loader, t_front_feat_loader = build_dataset(args, rank=rank)

    if args.mode == 'train':
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank, bert_tok=bert_tok, speaker_tok=speaker_tok, z_dicts=z_dicts,train_instr_data=train_instr_data,front_feat_loader=front_feat_loader,t_front_feat_loader=t_front_feat_loader)
    elif args.mode == 'valid':
        valid(args, train_env, val_envs, rank=rank, z_dicts=z_dicts,front_feat_loader=front_feat_loader, t_front_feat_loader=t_front_feat_loader)
    elif args.mode == 'extract_cfp_features':
        extract_cfp_features(args, train_env, z_dict=z_dicts, bert_tok=bert_tok)
            

if __name__ == '__main__':
    main()
