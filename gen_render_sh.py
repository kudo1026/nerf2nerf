import os
from datetime import datetime
from pathlib import Path

slurm_data_dir = Path('/share/data/ripl/slin/data')
slurm_project_dir = Path('/share/data/ripl/slin/projects/nerf2nerf')

local_data_dir = Path('/mnt/data')
local_project_dir = Path('/home/slin/projects/nerf2nerf')

logs_dir = slurm_data_dir / f'logs/n2n-{datetime.now().strftime("%m.%d_%H:%M:%S")}'
scripts_dir = Path('scripts/render')
sif_path = Path('/share/data/ripl/fjd/singularity_images/n2n_latest.sif')
os.makedirs(scripts_dir, exist_ok=True)

with open(scripts_dir / 'max_planck.sh', 'w') as f1:
    f1.write(f'mkdir -p {logs_dir}\n\n')
    for split in ['a', 'b']:
        for i in range(48):
            f1.write(f'sbatch -p ripl-gpu -c1 -C 2080ti -o {logs_dir}/{split}_{i}.out --wrap "singularity exec --nv --bind {slurm_project_dir}:{local_project_dir} --bind {slurm_data_dir}:{slurm_data_dir} {sif_path} bash {local_project_dir}/{scripts_dir}/{split}_{i}.sh"\n\n')
            with open(scripts_dir / f'{split}_{i}.sh', 'w') as f2:
                f2.write(f'cd {local_project_dir}/\n\n')
                f2.writelines([
                    'python3 render.py \\\n',
                    f'\t--a_or_b {split} \\\n',
                    f'\t--i {i} \\\n',
                    '\n',
                ])
