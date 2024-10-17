export WANDB_MODE=offline

# This script includes training, rendering (mesh extraction), and evaluation. You can modify according to your requirement.
# For quick evaluation, just comment the training and rendering parts and modify the data path accordingly.
# More arguments of train.py:
# --no_spike (disable the global FIF neuron)
# --no_cut (disable local FIF neruons)


n="dtu/release"
scale=0.01
declare -a s_array=("24" "37" "40" "55" "63" "65" "69" "83" "97" "105" "106" "110" "114" "118" "122")

for s in "${s_array[@]}"; do
    echo "Processing scene: dtu_${s}"
    # training
    python3 ./train.py \
        -s data/2ddtu/scan${s} \
        -m output/${n}_${s} \
        --exp_name "${n}_${s}" \
        --scale ${scale} \
        -r 2 \
        --lambda_smooth 1 \
        --ld_opa 0.0002 \
        --lambda_dist 1000.0 \
        --depth_ratio 1 \
        --mask_normal 1 
    # rendering (mesh extraction)
    python3 ./render_tsdf.py -m output/${n}_${s} --depth_ratio 1 --quiet
    # evaluation
    python3 ./script/eval_dtu/evaluate_single_scene.py --mask_dir data/2ddtu \
        --DTU dtu_eval \
        --scan_id ${s} \
        --input_mesh output/${n}_${s}/train/ours_30000/mesh/fused.ply \
        --output_dir cull_mesh/${s}

    python ./script/eval_dtu/eval.py --data cull_mesh/${s}/culled_mesh.ply \
         --scan ${s} --mode mesh \
         --dataset_dir dtu_eval \
         --vis_out_dir output/${n}_${s}/train/ours_30000/mesh

done
