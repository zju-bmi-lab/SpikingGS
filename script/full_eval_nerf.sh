export WANDB_MODE=offline

# This script includes training, rendering (mesh extraction), and evaluation. You can modify according to your requirement.
# For quick evaluation, just comment the training and rendering parts and modify the data path accordingly.
# More arguments of train.py:
# --no_spike (disable the global FIF neuron)
# --no_cut (disable local FIF neruons)


n="nerf/release"
scale=0.02
declare -a s_array=("lego" "hotdog" "materials" "ship" "drums" "mic" "chair"  "ficus" )

for s in "${s_array[@]}"; do
    echo "Processing scene: ${s}"
    # training
    python3 ./train.py \
        -s data/nerf_synthetic/${s} \
        -m output/${n}_${s} \
        --exp_name "${n}_${s}" \
        --scale ${scale} \
        --lambda_smooth 0.01 \
        --lambda_normal 0.05 \
        --eval 
    # rendering (mesh extraction)
    python3 ./render_tsdf.py -m output/${n}_${s} --skip_test --quiet
    # evaluation
    python ./eval_nerf.py --data_dir blender_eval/nerf_synthetic \
        --mesh_path output/${n}_${s}/train/ours_30000/mesh/fused_full.ply \
        --scene ${s} 
        #--resolution 400 # downsampling for faster evaluation

done
