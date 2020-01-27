# same as run_active_learn.sh
# but with less epochs training
# mostly for quick debugging.
EPOCHS=1
LR=0.001
GAMMA=0.1
INIT_SIZE=10
AL_BSIZE=100
SAMPLE_METHOD=dbal_bald
DROOT=data/mnist_easy
DNAME=mnist
OUT_DIR=output/
MAX_EPISODES=10
DROPOUT_ITR=5
CUDA_VISIBLE_DEVICES=0 python active_learn.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dataset-root $DROOT \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES \
                      --dropout-iterations $DROPOUT_ITR
