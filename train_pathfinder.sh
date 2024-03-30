export CUDA_VISIBLE_DEVICES=4

python -m train_pathfinder --num_epochs 2000 --lr 0.001 --seed 18 --weight_decay 0.01 --exp_name lra_pathfinder_lily --model_str LIN,128@TRAN,128,4,256,0.0,0.0,0.1,0.2,lily,hadamard,leakyrelu,LN,postnorm,6@REGH,2 --batch_size 32 --model_name "model" --accumulate 4