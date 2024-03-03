export CUDA_VISIBLE_DEVICES=3

python -m train_peptides_struct --num_epochs 3000 --lr 0.001 --seed 18 --weight_decay 0.1 --exp_name peptide_struct_onehot_16layers --model_str LIN,512@TRAN,512,8,40,0.1,maxout,cat,leakyrelu,16@REGH,11 --batch_size 16 --model_name "model" --accumulate 1