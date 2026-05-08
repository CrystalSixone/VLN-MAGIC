name=20260508_rxr_magic_l_valid
DATA_ROOT=../datasets

train_alg=dagger

ft_dim=768
features=clip768

ngpus=1
seed=0

outdir=${DATA_ROOT}/RxR/

student_model_type=magic_l

student_resume_file=${DATA_ROOT}/RxR/navigator/MAGIC_L/ckpts/best_val_unseen.pt
student_backdoor_file=${DATA_ROOT}/RxR/navigator/MAGIC_L/logs/backdoor/backdoor_update_features.tsv
student_frontdoor_file=${DATA_ROOT}/RxR/navigator/MAGIC_L/logs/frontdoor/frontdoor_update_features.tsv

flag="--root_dir ${DATA_ROOT}
      --dataset rxr
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --name ${name}   
      --mode valid

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy ndtw
      --train_alg ${train_alg}
      
      --max_action_len 28
      --max_instr_len 250

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
      --do_add_method door
      --z_instr_update

      --do_front_txt
      --do_front_img
      --do_front_his
      --front_n_clusters 24

      --student_model_type ${student_model_type}

      --student_resume_file ${student_resume_file}
      --s_frontdoor_dict_file ${student_frontdoor_file}
      --s_backdoor_dict_file ${student_backdoor_file}
      "

# valid
CUDA_VISIBLE_DEVICES='5' python r2r/main_nav.py $flag \
      --submit