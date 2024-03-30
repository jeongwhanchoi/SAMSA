export CUDA_VISIBLE_DEVICES=7

python -m train_coco --num_epochs 3000 --lr 0.001 --seed 18 --weight_decay 0.1 --exp_name coco_lily --model_str LIN,160@TRAN,160,16,150,0.3,0.1,0.0,0.2,lily,none,leakyrelu,LN,postnorm,8@LIN,81 --batch_size 32 --model_name "model" --accumulate 1