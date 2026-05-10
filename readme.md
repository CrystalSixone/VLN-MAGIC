# MAGIC: Meta-Ability Guided Interactive Chain-of-Distillation for Effective-and-Efficient Vision-and-Language Navigation

## Abstract

Despite the remarkable developments of recent large models in embodied Artificial Intelligence (E-AI), their application in robotics is hampered by their excessive parameter sizes and computational demands. Towards the Vision-and-Language Navigation (VLN) task, a core task in E-AI, this paper reveals the great potential of using knowledge distillation for obtaining lightweight student models by proposing a Meta-Ability Guided Interactive Chain-of-distillation (MAGIC) method. Specifically, a Meta-Ability Knowledge Distillation (MAKD) framework is proposed for decoupling and refining agents’ meta-abilities. A Meta-Knowledge Randomization Weighting (MKRW) and a Meta-Knowledge Transferable Determination (MKTD) module are incorporated to adjust aggregation weights at the meta-ability and sample levels respectively. Move beyond the traditional one-step unidirectional distillation, an Interactive Chain-of-Distillation (ICoD) strategy is proposed to allow students to give feedback to teachers, forming a new multi-step co-evolution pipeline. Remarkably, on the R2R test-unseen-public leaderboard, our smallest model, MAGIC-S, with only 5% of the teacher's size, outperforms all previous methods under the same training data. Additionally, our largest model, MAGIC-L, surpasses the previous SoTA by 5.84% in SPL and 3.18% in SR. Furthermore, a new dataset was collected and annotated from our living environments, where MAGIC-S demonstrated superior performance and real-time efficiency.

## Repository layout

| Path | Role |
|------|------|
| `pretrain_src/` | MAGIC pre-training (`run_r2r_magic.sh`, `run_rxr_magic.sh`) |
| `map_nav_src/` | Fine-tuning and validation (`r2r/main_nav.py`, Fairseq-style trainer, `scripts/*.sh`) |
| `datasets/` | Data, features, checkpoints, and logs (not fully tracked in git; see below) |

Fine-tuning scripts assume the current working directory is `map_nav_src/` (they call `python r2r/main_nav.py` with `DATA_ROOT=../datasets`).

## Setup

### 1. Requirements and installation

1. **Matterport3D Simulator** — Install from the official [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) repository.

2. **Python dependencies** — From the repository root:

```bash
pip install -r requirements.txt
```

Match the pinned versions when loading released checkpoints (notably `torch==1.9.0`, `transformers==4.34.1`).

3. **Download resources**

   - **Datasets, features, pre-trained weights:** [Hugging Face — crystal61/VLN-MAGIC](https://huggingface.co/crystal61/VLN-MAGIC/tree/main).
   - **METER pre-training (optional):** For METER-based pre-training, download `meter_clip16_224_roberta_pretrain.ckpt` from the [METER](https://github.com/zdou0830/METER) repository.
   - **EnvEdit weights (optional):** From [EnvEdit](https://github.com/jialuli-luka/EnvEdit).
   - **RoBERTa tokenizer:** If Hugging Face Hub access is limited, download [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base/tree/main) and place it under `datasets/pretrained/roberta`.

   Example layout for `datasets/` (adjust names to match your release archives):

```
datasets
├── R2R
│   ├── annotations
│   │   ├── pretrain_map
│   │   └── RxR
│   ├── connectivity
│   ├── features
│   ├── speaker
│   ├── navigator
│   ├── pretrain
│   ├── test
│   └── id_paths.json
├── RxR
│   ├── navigator
│   ├── pretrain
│   └── test
├── EnvEdit
└── pretrained
    ├── METER
    └── roberta
```

### 2. Pre-training

Run from `pretrain_src/` (edit `DATA_ROOT`, `CUDA_VISIBLE_DEVICES`, and GPU counts in the shell files as needed):

```bash
cd pretrain_src
bash run_r2r_magic.sh   # R2R
bash run_rxr_magic.sh   # RxR
```

### 3. Confounder feature extraction (Optional, you could use our provided features directly)

Follow [VLN-GOAT](https://github.com/CrystalSixone/VLN-GOAT) for confounder-related feature extraction used in this line of work.

### 4. Fine-tuning (knowledge distillation)

All navigation training scripts live under `map_nav_src/scripts/`. From `map_nav_src/`:

```bash
cd map_nav_src
bash scripts/run_r2r_kdl_train.sh
```

**ICoD (Interactive Chain-of-Distillation)** — second-stage co-evolution training:

```bash
cd map_nav_src
bash scripts/run_r2r_kdl_train_ICoD.sh
```

Edit each script for your machine: `DATA_ROOT`, `CUDA_VISIBLE_DEVICES`, `ngpus`, and paths to teacher/student checkpoints and front-door/back-door feature TSVs.

### 5. Validation

Validation also runs from `map_nav_src/` and invokes `r2r/main_nav.py` with `--mode valid`.

**R2R — standard MAGIC variants (S / M / L / B):**

```bash
cd map_nav_src
bash scripts/run_r2r_kdl_valid_magicS.sh
bash scripts/run_r2r_kdl_valid_magicM.sh
bash scripts/run_r2r_kdl_valid_magicL.sh
bash scripts/run_r2r_kdl_valid_magicB.sh
```

**R2R — ICoD checkpoints:**

```bash
cd map_nav_src
bash scripts/run_r2r_kdl_valid_ICoD.sh
```

**RxR** — analogous scripts under the same folder, e.g. `run_rxr_kdl_valid.sh`, `run_rxr_kdl_valid_magic*.sh`, `run_rxr_kdl_valid_ICoD*.sh`.

Many validation scripts pass `--submit` so that the **test-unseen** split is evaluated when you intend to produce leaderboard-style submissions. Toggle or remove `--submit` in the script if you only need val-seen / val-unseen.

## BibTeX

If you find our work useful in your research, please consider citing:

```bibtex
@article{Wang2026MAGIC,
  author  = {Wang, Liuyi and He, Zongtao and Shen, Mengjiao and Yang, Jingwei and Liu, Chengju and Chen, Qijun},
  title   = {{MAGIC}: Meta-Ability Guided Interactive Chain-of-Distillation for Effective-and-Efficient Vision-and-Language Navigation},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026}
}
```
