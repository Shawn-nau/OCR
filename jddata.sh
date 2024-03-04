
model_name=MLPQR
loss=Omni_cost_loss

root_path_name=.exp/
data_path_name=processed_df.csv

random_seed=20240205

for set in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 
do
python -u run_exp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model $model_name \
    --loss $loss \
    --dim_embedding 3 \
    --hidden_size1 88 \
    --hidden_size2 44 \
    --train_epochs 100 \
    --s $set\
    --itr 1 --batch_size 5120 --learning_rate 0.001
done 


