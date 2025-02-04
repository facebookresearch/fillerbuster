dataset_dir="/mnt/home/ethanjohnweber/data/nerfbusters-dataset"
render_dir="/mnt/home/ethanjohnweber/data/nerfbusters-renders"
datasets="aloe art car century flowers garbage picnic pipe plant roses table"
methods="mvinpaint gsplat cat3d nerfiller"
extra=""
for dataset in $datasets; do
  for method in $methods; do
    sbatch mvinpaint/scripts/experiments/complete_scene.sh $dataset_dir $render_dir $dataset $method $extra
  done
done
