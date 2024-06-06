# MAGIC: Meta-Ability Guided Interactive Chain-of-Distillation for Effective-and-Efficient Vision-and-Language Navigation

## Abstract
Despite the remarkable developments of recent large models in embodied Artificial Intelligence (E-AI), their application in robotics is hampered by their excessive parameter sizes and computational demands. Towards the Vision-and-Language Navigation (VLN) task, a core task in E-AI, this paper reveals the great potential of using knowledge distillation for obtaining lightweight student models by proposing a Meta-Ability Guided Interactive Chain-of-distillation (MAGIC) method. Specifically, a Meta-Ability Knowledge Distillation (MAKD) framework is proposed for decoupling and refining agents’ meta-abilities. A Meta-Knowledge Randomization Weighting (MKRW) and a Meta-Knowledge Transferable Determination (MKTD) module are incorporated to adjust aggregation weights at the meta-ability and sample levels respectively. Move beyond the traditional one-step unidirectional distillation, an Interactive Chain-of-Distillation (ICoD) strategy is proposed to allow students to give feedback to teachers, forming a new multi-step co-evolution pipeline. Remarkably, on the R2R test-unseen-public leaderboard, our smallest model, MAGIC-S, with only 5% of the teacher's size, outperforms all previous methods under the same training data. Additionally, our largest model, MAGIC-L, surpasses the previous SoTA by 5.84% in SPL and 3.18% in SR. Furthermore, a new dataset was collected and annotated from our living environments, where MAGIC-S demonstrated superior performance and real-time efficiency.


## Setup Instructions

### 1. Requirements and Installation

1. **Install MatterPort3D Simulator:** Start by installing the MatterPort3D simulator from the official [repository](https://github.com/peteanderson80/Matterport3DSimulator).

2. **Install Python Dependencies:** Run the following command to install the necessary Python packages. Make sure to match the versions in `requirements.txt` to avoid compatibility issues, particularly when loading pre-trained weights for fine-tuning.
    ```setup
    pip install -r requirements.txt
    ```
3. **Download Resources**:
    1. **Datasets and Features:**: Links will be updated soon.
    2. **Pre-trained Weights**: Links will be updated soon.
    3. **METER Pre-training (Optional):** If you wish to pre-train GOAT using METER, download the model `meter_clip16_224_roberta_pretrain.ckpt` from [here](https://github.com/zdou0830/METER).
    4. **EnvEdit Weights (Optional)**: Available [here](https://github.com/jialuli-luka/EnvEdit).
    5. **RoBERTa Tokenizer**: If direct access to Hugging Face models is restricted, manually download `roberta-base` from [Hugging Face](https://huggingface.co/FacebookAI/roberta-base/tree/main) and store it locally under `datasets/pretrained/roberta`.

    Ensure your `datasets` directory follows this structure:
    ```
    datasets
    ├── R2R
    │   ├── annotations
    │   │   ├──pretrain_map
    │   │   └──RxR
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

To pre-train the model, navigate to the pre-training source directory and execute the provided shell script. Replace r2r with the desired dataset name as needed.
```pre-train
cd pretrain_src
bash run_r2r_magic.sh
```

### 3. Confounder Feature Extraction
Please refer to [GOAT's repository](https://github.com/CrystalSixone/VLN-GOAT) for confounder feature extraction.

### 4. Fine-tuning
To fine-tune the model, use the command below:
``` fine-tune
cd map_nav_src
bash scripts/run_r2r.sh
```

### 5. Validation
For model validation, execute the following:
``` valid
cd map_nav_src
bash scripts/run_r2r_valid.sh
```

----
Since this article is still under review, we have omitted the model files. We plan to gradually release the training and validation code in the future.

Thank you for your understanding and support!