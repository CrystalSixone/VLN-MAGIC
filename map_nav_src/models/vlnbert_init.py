import torch
from transformers import AutoModel


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    elif args.tokenizer == 'roberta':
        cfg_name = 'roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(args, config=None, role='student'):
    from transformers import PretrainedConfig
    from models.vilmodel_GOAT import GlocalTextPathNavCMT
    
    if role == 'student':
        model_name_or_path = args.student_bert_ckpt_file
    elif role == 'teacher':
        model_name_or_path = args.teacher_bert_ckpt_file # useful only if teacher is being trained
    new_ckpt_weights = {}
    if model_name_or_path == 'bert':
        tmp = AutoModel.from_pretrained('bert-base-uncased')
        for param_name, param in tmp.named_parameters():
            # new_ckpt_weights[param_name] = param
            if 'bert.encoder.layer' in param_name:
                param_name = param_name.replace('bert.encoder.layer', 'bert.lang_encoder.layer')
                new_ckpt_weights[param_name] = param
            else:
                new_ckpt_weights[param_name] = param
        del tmp
    elif model_name_or_path == 'meter':
        try:
            tmp = torch.load('../datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
        except Exception:
            tmp = torch.load('datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
        tmp = tmp['state_dict']
        for param_name, param in tmp.items():
            if 'text_transformer.embeddings' in param_name:
                param_name = param_name.replace('text_transformer.', 'bert.')
                new_ckpt_weights[param_name] = param
            elif 'text_transformer.encoder' in param_name:
                param_name = param_name.replace('text_transformer.encoder', 'bert.lang_encoder')
                new_ckpt_weights[param_name] = param
            elif 'cross_modal_image_layers' in param_name:
                param_name1 = param_name.replace('cross_modal_image_layers', 'bert.local_encoder.encoder.crossattention')
                param_name2 = param_name.replace('cross_modal_image_layers', 'bert.global_encoder.encoder.crossattention')
                new_ckpt_weights[param_name1] = new_ckpt_weights[param_name2] = param
            else:
                new_ckpt_weights[param_name] = param
        del tmp
    elif model_name_or_path is not None:
        # pretrain model (path)
        model_name = None
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if k.startswith('vln_bert'):
                k = 'bert' + k[8:]
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            # if 'kdl_img_w' in k or 'kdl_avg_img_w' in k: # !!!
            #     new_ckpt_weights['bert.' + k[20:]] = v
            elif 'tim' in k or 'temperature' in k:
                if 'self_encoder' not in k:
                    new_ckpt_weights['bert.' + k] = v
                else:
                    new_ckpt_weights[k] = v
            else:
                new_ckpt_weights[k] = v
            
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    elif args.tokenizer == 'roberta':
        # cfg_name = 'roberta-base'
        cfg_name = 'datasets/pretrained/roberta'
    else:
        cfg_name = 'bert-base-uncased'
    try:
        vis_config = PretrainedConfig.from_pretrained(cfg_name)
    except Exception:
        cfg_name = '../' + cfg_name
        vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    elif args.tokenizer == 'roberta':
        assert vis_config.type_vocab_size == 1
    
    # Convert args to a dictionary and update vis_config with these values
    args_dict = vars(args)
    for key, value in args_dict.items():
        setattr(vis_config, key, value)
    
    vis_config.max_action_steps = 100
    vis_config.obj_loc_size = 3
    vis_config.obj_name_vocab_size = 45
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.update_lang_bert = not args.fix_lang_embedding 
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    
    # METER param
    # Text Setting
    vis_config.type_vocab_size = 1
    vis_config.max_position_embeddings = 514
    vis_config.vocab_size = 50265 
    vis_config.mlm_prob = 0.15
    vis_config.draw_false_text = 0
    vis_config.attention_probs_dropout_prob = 0.1

    # knowledge distillation
    if role == 'student':
        vis_config.num_top_layer = args.student_num_x_layers # cross-attention
        vis_config.num_hidden_layers = args.student_num_l_layers # language BERT
        vis_config.num_l_layers = args.student_num_l_layers 
        vis_config.num_pano_layers = args.student_num_pano_layers 
        vis_config.num_x_layers = args.student_num_x_layers 
        vis_config.hidden_size = args.student_hidden_size # 768
        vis_config.num_attention_heads = vis_config.hidden_size // 64
        vis_config.mlp_ratio = args.student_mlp_ratio 
        vis_config.intermediate_size = vis_config.hidden_size * vis_config.mlp_ratio
        vis_config.teacher_hidden_size = args.teacher_hidden_size
        vis_config.role = 'student'
    elif role == 'teacher':
        vis_config.num_top_layer = args.teacher_num_x_layers # cross-attention
        vis_config.num_hidden_layers = args.teacher_num_l_layers # language BERT
        vis_config.num_l_layers = args.teacher_num_l_layers 
        vis_config.num_pano_layers = args.teacher_num_pano_layers 
        vis_config.num_x_layers = args.teacher_num_x_layers 
        vis_config.hidden_size = args.teacher_hidden_size # 768
        vis_config.num_attention_heads = vis_config.hidden_size // 64
        vis_config.mlp_ratio = args.teacher_mlp_ratio # 4
        vis_config.intermediate_size = vis_config.hidden_size * vis_config.mlp_ratio
        vis_config.student_hidden_size = args.student_hidden_size
        vis_config.role = 'teacher'

    vis_config.name = 'R2R'
    if args.dataset == 'reverie':
        vis_config.name = 'REVERIE'
        vis_config.use_obj_name = True
        
    elif args.dataset == 'soon':
        vis_config.name ='SOON'
        vis_config.use_obj_name = False
        
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights,
        ignore_mismatched_sizes=True) # NOTE: this is for loading student models as the teacher models
        
    return visual_model
