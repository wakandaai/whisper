#!/usr/bin/env python3
"""Generate run.sh scripts for each ablation experiment."""

import os

EXPERIMENTS = [
    "sw_asr",
    "sw_to_en",
    "sw_to_fr",
    "sw_to_en_fr",
    "sw_asr_ast_en",
    "sw_asr_ast_fr",
    "sw_asr_ast_en_fr",
    "rw_asr",
    "rw_to_en",
    "rw_to_fr",
    "rw_to_en_fr",
    "rw_asr_ast_en",
    "rw_asr_ast_fr",
    "rw_asr_ast_en_fr",
]

TEMPLATE = """#!/usr/bin/env bash
source /ocean/projects/cis250145p/gichamba/miniconda3/etc/profile.d/conda.sh
conda activate whisper
echo "Active environment: $CONDA_DEFAULT_ENV"
module load ffmpeg
python3 whisper/trainer.py --config iwslt/train_conf/cv_ablations/{exp}.yaml
"""

OUTPUT_DIR = "run_scripts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for exp in EXPERIMENTS:
    script = TEMPLATE.format(exp=exp)
    path = os.path.join(OUTPUT_DIR, f"run_{exp}.sh")
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)  # make executable

print(f"Generated {len(EXPERIMENTS)} scripts in {OUTPUT_DIR}/")