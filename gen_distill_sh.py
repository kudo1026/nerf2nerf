import os
from datetime import datetime
from pathlib import Path

slurm_data_dir = Path('/share/data/ripl/slin/data')
slurm_project_dir = Path('/share/data/ripl/slin/projects/nerf2nerf')
local_project_dir = Path('/home/slin/projects/nerf2nerf')

logs_dir = slurm_data_dir / f'logs/n2n-distill-{datetime.now().strftime("%m.%d_%H:%M:%S")}'
scripts_dir = Path('scripts/distill')
sif_path = Path('/share/data/ripl/fjd/singularity_images/n2n_latest.sif')
os.makedirs(scripts_dir, exist_ok=True)

with open(scripts_dir / f'distill.sh', 'w') as f1:
    f1.write(f'mkdir -p {logs_dir}\n\n')
    for scene_no in range(1, 7):
        for split in ['a', 'b']:
            f1.write(f'sbatch -p ripl-gpu -c1 -C 2080ti -o {logs_dir}/{scene_no}{split}.out --wrap "singularity exec --nv --bind {slurm_project_dir}:{local_project_dir} --bind {slurm_data_dir}:{slurm_data_dir} {sif_path} bash {local_project_dir}/{scripts_dir}/{scene_no}{split}.sh"\n\n')
            with open(scripts_dir / f'{scene_no}{split}.sh', 'w') as f2:
                f2.write(f'cd {local_project_dir}/\n\n')
                f2.writelines([
                    'python3 distill.py \\\n',
                    f'\t--scene_no {scene_no} \\\n',
                    f'\t--a_or_b {split} \\\n',
                    f'\t--step 2 \\\n',
                    '\n',
                ])
