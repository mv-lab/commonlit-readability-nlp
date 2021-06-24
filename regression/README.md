# Best models so far

## Roberta-large
LB 0.465 <br>
CV around 0.47-0.483 across 5 folds. Reran some folds which were high RMSE with different LR. Some folds did worse on pretrained ITPT model so removed it. 

`python pytorch_bert.py -model roberta-large  -model_dir roberta_l_0624 -max_len 250 -batch_size 8 -scheduler cosine -lr 2e-5 -epochs 10 -gpu 0 1 -wd 0.01 -hidden_size 1024 -eval_steps 10 -use_dp -use_dropout 0.1 -use_hidden mean  -fc_size 4096 -pretrained_model roberta_large_pretrain_0609`


## Deberta-large
LB 0.471

`python pytorch_bert.py -model microsoft/deberta-large -model_dir deberta_l_0616 -max_len 250 -batch_size 4 -test_batch_size 4 -gpu 0 1 -wd 0.01 -hidden_size 1024 -use_dp -eval_steps 10 -epochs 5 -use_hidden mean -fc_size 4096 -use_single_fc -use_dropout 0.3 -scheduler cosine -lr 1e-5 -pretrained_model deberta_large_pretrain_0616`

