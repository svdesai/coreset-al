EPOCHS=10
LR=1.0
GAMMA=0.1
INIT_SIZE=2000
AL_BSIZE=1000
SAMPLE_METHOD=coreset_better
DROOT=data/mnist_easy
DNAME=mnist
OUT_DIR=output/
MAX_EPISODES=10
CUDA_VISIBLE_DEVICES=0 python active_learn.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dataset-root $DROOT \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES
