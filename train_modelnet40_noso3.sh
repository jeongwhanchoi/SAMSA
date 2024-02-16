export CUDA_VISIBLE_DEVICES=1

python -m train_modelnet40 --num_epochs 3000 --num_points 1024 --lr 0.001 --seed 18 --weight_decay 0.1 --input_mask False --exp_name modelnet_noso3 --uniform True --use_normal True --model_str LIN,256@TRAN,256,8,256,0.1,maxout,cat,leakyrelu,8@CLFH,40,256,512,0.1 --batch_size 64 --model_name "model" --accumulate 1