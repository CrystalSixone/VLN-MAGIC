name=r2r_Student_magicM_Teacher_magicB_ICoD_valid
DATA_ROOT=../datasets

train_alg=dagger

ft_dim=768
features=clip768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/

student_model_type=magic_m
teacher_model_type=magic_b

teacher_resume_file=${DATA_ROOT}/R2R/navigator/MAGIC_M_ICoD/ckpts/teacher_best_val_unseen.pt
teacher_backdoor_file=${DATA_ROOT}/R2R/navigator/MAGIC_M_ICoD/logs/backdoor/teacher_backdoor_update_features.tsv
teacher_frontdoor_file=${DATA_ROOT}/R2R/navigator/MAGIC_M_ICoD/logs/frontdoor/teacher_frontdoor_update_features.tsv

student_resume_file=${DATA_ROOT}/R2R/navigator/MAGIC_M_ICoD/ckpts/best_val_unseen.pt
student_backdoor_file=${DATA_ROOT}/R2R/navigator/MAGIC_M_ICoD/logs/backdoor/backdoor_update_features.tsv
student_frontdoor_file=${DATA_ROOT}/R2R/navigator/MAGIC_M_ICoD/logs/frontdoor/frontdoor_update_features.tsv


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
      --do_add_method door
      --z_instr_update

      --do_front_txt
      --do_front_img
      --do_front_his
      --front_n_clusters 24

      --train_kdl

      --student_model_type ${student_model_type}
      --teacher_model_type ${teacher_model_type}

      --teacher_resume_file ${teacher_resume_file}
      --t_frontdoor_dict_file ${teacher_frontdoor_file}
      --backdoor_dict_file ${teacher_backdoor_file}

      --student_resume_file ${student_resume_file}
      --s_frontdoor_dict_file ${student_frontdoor_file}
      --s_backdoor_dict_file ${student_backdoor_file}
      "

# valid
CUDA_VISIBLE_DEVICES='3' python r2r/main_nav.py $flag 

# submit
# CUDA_VISIBLE_DEVICES='3' python r2r/main_nav.py $flag \
      # --submit