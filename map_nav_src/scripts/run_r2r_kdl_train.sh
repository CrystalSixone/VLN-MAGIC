#!/bin/bash
name=r2r_magic-b_train
DATA_ROOT=../datasets

train_alg=dagger

ft_dim=768
features=clip768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/

speaker_envedit_file=${DATA_ROOT}/R2R/speaker/transpeaker_r2r/state_dict/best_both_bleu
augdir=${DATA_ROOT}/R2R/annotations/prevalent_aug_train_enc.json

teacher_file=${DATA_ROOT}/R2R/navigator/GOAT/ckpts/best_val_unseen
teacher_backdoor_file=${DATA_ROOT}/R2R/navigator/GOAT/logs/backdoor/update_instr_z_dict.tsv
teacher_frontdoor_file=${DATA_ROOT}/R2R/navigator/GOAT/logs/frontdoor/z_front_feature.tsv

student_B_pretrain_file=${DATA_ROOT}/R2R/pretrain/MAGIC_B
student_M_pretrain_file=${DATA_ROOT}/R2R/pretrain/MAGIC_M
student_S_pretrain_file=${DATA_ROOT}/R2R/pretrain/MAGIC_S

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --name ${name}   
      --mode train

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 4e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.5
      --dropout 0.1
      
      --gamma 0.
      
      --cat_file ../datasets/R2R/annotations/category_mapping.tsv

      --use_transpeaker
      --speaker ${speaker_envedit_file}
      --accumulateGrad

      --use_aug_env
      --aug ${augdir}

      --do_back_txt
      --do_back_img
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door
      --z_instr_update

      --do_front_txt
      --do_front_img
      --do_front_his
      --front_n_clusters 24

      --train_kdl

      --teacher_resume_file ${teacher_file}
      --kdl_temperature 2
      --kdl_alpha 0.5
      --t_frontdoor_dict_file ${teacher_frontdoor_file}
      --backdoor_dict_file ${teacher_backdoor_file}

      --teacher_hidden_size 768
      --teacher_num_l_layers 6
      --teacher_num_pano_layers 2
      --teacher_num_x_layers 3
      --teacher_mlp_ratio 4

      --student_num_l_layers 6
      --student_num_x_layers 3
      --student_num_pano_layers 2
      --student_hidden_size 384
      --student_mlp_ratio 4

      --kdl_adaptive_ability_weight
      --kdl_adaptive_ability_weight_type RW
      --aw_update_iter 100
      --rw_temp 4
      --teacher_sample_hard_mining
      --t_sample_preprocess exp
      --t_sample_preprocess_exp_decay 0.7

      --use_lr_sch
      --use_warm_up
      "

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --tokenizer roberta \
      --student_bert_ckpt_file ${student_B_pretrain_file}