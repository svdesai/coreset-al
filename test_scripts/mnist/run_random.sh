EPOCHS=50
LR=0.001
GAMMA=0.1
INIT_SIZE=50
AL_BSIZE=50
SAMPLE_METHOD=random
DROOT=../../data/mnist_easy
DNAME=mnist
OUT_DIR=../../output/
MAX_EPISODES=20
CUDA_VISIBLE_DEVICES=0 python ../../active_learn.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dataset-root $DROOT \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES
