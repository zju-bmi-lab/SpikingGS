export WANDB_MODE=offline

# This script includes training, rendering (mesh extraction), and evaluation. You can modify according to your requirement.
# For quick evaluation, just comment the training and rendering parts and modify the data path accordingly.
# More arguments of train.py:
# --no_spike (disable the global FIF neuron)
# --no_cut (disable local FIF neruons)


n="dex/release" 
scale=1.0
declare -a s_array=("mount" "pipe" "pawn" "turbine" "up" "side" )

for s in "${s_array[@]}"; do
    echo "Processing scene: ${s}"
    # training
    python3 ./train.py \
        -s data/semi/${s} \
        -m output/${n}_${s} \
        --exp_name "${n}_${s}" \
        --scale ${scale} \
        --lambda_smooth 5 \
        --lambda_dist 50.0 \
        --lambda_normal 0.07 \
        --eval \
        -r 2 \
        --lambda_tv_d 1 \
        --depth_ratio 1 \
        --fil_width 0
    # rendering (mesh extraction)
    python3 ./render_tsdf.py -m output/${n}_${s} --skip_test --quiet --depth_ratio 1 
    # evaluation
    python ./eval_dex.py --data_dir blender_eval/dexnerf \
        --mesh_path output/${n}_${s}/train/ours_30000/mesh/fused.ply \
        --scene ${s} 
        #--resolution 400 # downsampling for faster evaluation

done


