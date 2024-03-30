export CUDA_VISIBLE_DEVICES=1

python -m train_text --num_epochs 2000 --lr 0.004 --seed 18 --weight_decay 0.01 --exp_name lra_text_lily --model_str LIN,128@TRAN,128,4,256,0.3,0.1,0.0,0.2,lily,hadamard,leakyrelu,LN,postnorm,4@REGH,2 --batch_size 32 --model_name "model" --accumulate 1