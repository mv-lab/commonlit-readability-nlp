# Best models so far

For all models:  Reran some folds which were high RMSE with different LR. Some folds did worse on pretrained ITPT model so removed it. 


## Funnel transformer large
LB 0.465 <br>

`python pytorch_bert.py -model funnel-transformer/large -max_len 250 -batch_size 4 -scheduler cosine -lr 1e-5 -epochs 5 -gpu 0 1 -wd 0.01 -hidden_size 1024 -eval_steps 50 -use_dp -use_dropout 0.3 -fc_size 1024 -model_dir funnel_l_0702`

Best RMSE in fold: 0 was: 0.4764 run command above <br> 
Best RMSE in fold: 1 was: 0.4870 eval 10, drop 0.1 lr 2e-5 wd 0.01.  0.487 with pretrained <br> 
Best RMSE in fold: 2 was: 0.4890 eval 10, drop 0.1 lr 2e-5 wd 0.01 <br> 
Best RMSE in fold: 3 was: 0.4810 run command above <br> 
Best RMSE in fold: 4 was: 0.4880 eval 10, drop 0.1 lr 2e-5 wd 0.01 <br> 



## Roberta-large
LB 0.465 <br>
CV around 0.47-0.483 across 5 folds.

`python pytorch_bert.py -model roberta-large  -model_dir roberta_l_0624 -max_len 250 -batch_size 8 -scheduler cosine -lr 2e-5 -epochs 5 -gpu 0 1 -wd 0.01 -hidden_size 1024 -eval_steps 10 -use_dp -use_dropout 0.1 -fc_size 1024 -pretrained_model roberta_large_pretrain_0609`


CV: <br>
roberta_l_0621_fold0/params.txt:Best RMSE in fold: 0 was: 0.4763 -- No pretrained model <br>
roberta_l_0621_fold1/params.txt:Best RMSE in fold: 1 was: 0.4843 <br>
roberta_l_0621_fold2/params.txt:Best RMSE in fold: 2 was: 0.4840 <br>
roberta_l_0621_fold3/params.txt:Best RMSE in fold: 3 was: 0.4719 <br>
roberta_l_0621_fold4/params.txt:Best RMSE in fold: 4 was: 0.4732 <br>



## Deberta-large
LB  0.466 

`python pytorch_bert.py -model microsoft/deberta-large -model_dir deberta_l_0627 -max_len 250 -batch_size 4 -test_batch_size 4 -gpu 0 1 -wd 0.01 -hidden_size 1024 -use_dp -eval_steps 35 -epochs 5 -fc_size 1024 -use_dropout 0.3 -scheduler cosine -lr 2e-5 -pretrained_model deberta_large_pretrain_0616`

CV: <br>
../deberta_l_0627/params.txt:Best RMSE in fold: 0 was: 0.4848 - post fix. 0.4758. lr 1e-5. eval 10. dropout 0.3 <br>
../deberta_l_0627/params.txt:Best RMSE in fold: 1 was: 0.4897 - post fix 0.4888. seed 42. lr 1e-5 step 20 <br>
../deberta_l_0627/params.txt:Best RMSE in fold: 2 was: 0.4845 <br>
../deberta_l_0627/params.txt:Best RMSE in fold: 3 was: 0.5059 - worse pretrained. Post fix 0.4778 removed pretrained model eval 20 wd 0.001 <br>
../deberta_l_0627/params.txt:Best RMSE in fold: 4 was: 0.4712


## Deberta-large
LB 0.471

`python pytorch_bert.py -model microsoft/deberta-large -model_dir deberta_l_0616 -max_len 250 -batch_size 4 -test_batch_size 4 -gpu 0 1 -wd 0.01 -hidden_size 1024 -use_dp -eval_steps 10 -epochs 5 -use_hidden mean -fc_size 4096 -use_single_fc -use_dropout 0.3 -scheduler cosine -lr 1e-5 -pretrained_model deberta_large_pretrain_0616`

CV: <br>
deberta_l_0617_fold0/params.txt:Best RMSE in fold: 0 was: 0.4679 <br>
deberta_l_0617_fold1/params.txt:Best RMSE in fold: 1 was: 0.4847 <br>
deberta_l_0617_fold2/params.txt:Best RMSE in fold: 2 was: 0.4777 <br>
deberta_l_0617_fold3/params.txt:Best RMSE in fold: 3 was: 0.4731 <br>
deberta_l_0617_fold4/params.txt:Best RMSE in fold: 4 was: 0.4867 <br>

## Electra-large
LB 0.468

`python pytorch_bert.py -model google/electra-large-discriminator -max_len 256 -batch_size 8 -test_batch_size 8 -gpu 0 1 -wd 0.01 -hidden_size 1024 -use_dp   -fc_size 1024 -use_dropout 0.1 -multisample_dropout -scheduler linear -lr 2e-5 -epochs 5 -eval_steps 50 -pretrained_model electra_large_pretrain_0620`


CV:  <br>
../electra_l_0620_folds0/params.txt:Best RMSE in fold: 0 was: 0.4751 <br>
../electra_l_0620_folds1/params.txt:Best RMSE in fold: 1 was: 0.4840 <br>
../electra_l_0620_folds2/params.txt:Best RMSE in fold: 2 was: 0.4736 <br>
../electra_l_0620_folds3/params.txt:Best RMSE in fold: 3 was: 0.4583 <br>
../electra_l_0620_folds4/params.txt:Best RMSE in fold: 4 was: 0.4693 <br>

## Electra-large
LB 0.471

`python pytorch_bert.py -model google/electra-large-discriminator -max_len 256 -batch_size 8 -test_batch_size 8 -gpu 0 1 -wd 0.01 -hidden_size 1024 -use_dp   -fc_size 1024 -use_dropout 0.1 -multisample_dropout -scheduler linear -lr 2e-5 -epochs 5 -eval_steps 50 -pretrained_model electra_large_pretrain_0620`


CV:  <br>
electra_l_0620_folds0/params.txt:Best RMSE in fold: 0 was: 0.4751 <br>
electra_l_0620_folds1/params.txt:Best RMSE in fold: 1 was: 0.4840 <br>
electra_l_0620_folds2/params.txt:Best RMSE in fold: 2 was: 0.4736 <br>
electra_l_0620_folds3/params.txt:Best RMSE in fold: 3 was: 0.4711 <br>
electra_l_0620_folds4/params.txt:Best RMSE in fold: 4 was: 0.4746 <br>
