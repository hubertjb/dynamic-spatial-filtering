##
# Train models on a 3-recording subset of PC18 and compare their robustness to
# channel corruption.
##

# Some shared parameters across experiments
SEED=87
N_JOBS=4
DETERMINISTIC=true
NUM_WORKERS=4

DATASET=pc18_debug
DROPOUT=0.5
BATCH_SIZE=64
LR=0.0004
N_EPOCHS=2
PATIENCE=2
WEIGHT_DECAY=0.001

# Make a directory for saving the results
BASE_DIR="./runs/${DATASET}"
DATE_TIME=$(date '+%d-%m-%y_%H-%M-%S')
SAVE_DIR="${BASE_DIR}/${DATE_TIME}"
mkdir -p $SAVE_DIR

# Train models: {vanilla, DSFd, DSFm-st} x {no denoising, autoreject, data augm}

for DSF_MLP_INPUT in vanilla dsfd dsfm_st
do
    for DENOISING in no_denoising autoreject data_augm
    do
        echo -e "\n####"
        echo "# $DSF_MLP_INPUT, $DENOISING"
        echo -e "####\n"
        python train.py \
            --seed $SEED \
            --n_jobs $N_JOBS \
            --deterministic $DETERMINISTIC \
            --num_workers $NUM_WORKERS \
            --save_dir $SAVE_DIR \
            --dataset $DATASET \
            --valid_size 0.2 \
            --test_size 0.2 \
            --random_state_valid $SEED \
            --random_state_test $SEED \
            --window_size_s 30 \
            --model stager_net \
            --dropout $DROPOUT \
            --dsf_type $DSF_MLP_INPUT \
            --dsf_n_out_channels None \
            --denoising $DENOISING \
            --lr $LR \
            --batch_size $BATCH_SIZE \
            --n_epochs $N_EPOCHS \
            --patience $PATIENCE \
            --weight_decay $WEIGHT_DECAY \
            --cosine_annealing True
    done
done

# Evaluate performance under noise

python evaluate_noise_robustness.py \
    --exp_dir $SAVE_DIR \
    --seed $SEED \
    --n_jobs $N_JOBS \
    --deterministic $DETERMINISTIC \
    --num_workers $NUM_WORKERS \
    --dataset $DATASET \
    --valid_size 0.2 \
    --test_size 0.2 \
    --random_state_valid $SEED \
    --random_state_test $SEED \
    --window_size_s 30
