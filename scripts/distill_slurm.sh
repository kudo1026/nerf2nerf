mkdir -p /share/data/ripl/slin/data/logs/distill_n2n

sbatch -p ripl-gpu -c1 -C 2080ti -o /share/data/ripl/slin/data/logs/distill_n2n/a.out --wrap "singularity exec --nv --bind /share/data/ripl/slin/projects/nerf2nerf:/home/slin/projects/nerf2nerf --bind /share/data/ripl/slin/data:/share/data/ripl/slin/data /share/data/ripl/fjd/singularity_images/nerfstudio_v0.1.15-hloc.sif bash /home/slin/projects/nerf2nerf/scripts/distill_a.sh"

sbatch -p ripl-gpu -c1 -C 2080ti -o /share/data/ripl/slin/data/logs/distill_n2n/b.out --wrap "singularity exec --nv --bind /share/data/ripl/slin/projects/nerf2nerf:/home/slin/projects/nerf2nerf --bind /share/data/ripl/slin/data:/share/data/ripl/slin/data /share/data/ripl/fjd/singularity_images/nerfstudio_v0.1.15-hloc.sif bash /home/slin/projects/nerf2nerf/scripts/distill_b.sh"
