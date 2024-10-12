python main_ind.py --dataname citeseer --epoch 100 --lr1 0.0005 --lambda_loss 1 --moving_average_decay 0.0 --hid_dim 1024 --temp 0.9 --lam 0.01 --num_MLP 0 --n_layers 1 --wd2 0.01
python main_ind.py --dataname pubmed --epoch 500 --lr1 0.0005 --lambda_loss 1 --moving_average_decay 0.0 --hid_dim 4096 --temp 0.7 --lam 0.5 --num_MLP 2 --n_layers 1 --wd2 0.01
python main_ind.py --dataname photo --epoch 700 --lr1 0.0005 --lambda_loss 1 --moving_average_decay 0.0 --hid_dim 2048 --temp 0.5 --lam 0.01 --num_MLP 1 --n_layers 2
