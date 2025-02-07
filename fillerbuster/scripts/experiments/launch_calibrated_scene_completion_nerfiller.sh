dataset_dir="/mnt/home/ethanjohnweber/data/nerfiller-dataset"
render_dir="/mnt/home/ethanjohnweber/data/nerfiller-renders"
datasets="bear billiards boot cat drawing dumptruck norway office turtle"
methods="fillerbuster-no-new-views nerfiller-no-new-views fillerbuster-no-new-views-no-normals nerfiller-no-new-views-no-normals mask gsplat"
extra="--path nerfiller"
for dataset in $datasets; do
  for method in $methods; do
    sbatch fillerbuster/scripts/experiments/slurm_calibrated_scene_completion.sh $dataset_dir $render_dir $dataset $method "$extra"
  done
done
