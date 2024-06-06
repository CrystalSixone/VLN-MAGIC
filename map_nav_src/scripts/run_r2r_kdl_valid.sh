name=r2r_magic-b_valid
DATA_ROOT=../datasets

train_alg=dagger

ft_dim=768
features=clip768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --name ${name}   
      --mode valid

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 16
      --lr 4e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.3
      --dropout 0.1
      
      --gamma 0.
      
      --cat_file ../datasets/R2R/annotations/category_mapping.tsv
      --adaptive_pano_fusion

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

      --teacher_resume_file ${teacher_resume_file}
      --kdl_temperature 2
      --kdl_alpha 0.5
      --t_frontdoor_dict_file ${teacher_frontdoor_file}
      --backdoor_dict_file ${teacher_backdoor_file}

      --kdl_feat_loss mse
      --kdl_attn_loss mse
      --kdl_logit_loss kd
      --kdl_dkd_alpha 1
      --kdl_dkd_beta 4

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
      --rw_temp 4
      --teacher_sample_hard_mining
      --t_sample_preprocess exp
      --t_sample_preprocess_exp_decay 0.7

      --s_frontdoor_dict_file ${student_frontdoor_file}
      --s_backdoor_dict_file ${student_backdoor_file}

      --submit
      "

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --tokenizer roberta \
      --student_resume_file ${student_resume_file}