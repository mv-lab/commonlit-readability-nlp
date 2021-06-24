
config = {
          'pretrained_model_name_or_path': '/content/drive/MyDrive/commonlit/models/mlm_roberta_large',
          'tokenizer_name': 'roberta-large',
          'curr_best': 1e15,
          'partial_eval': 4,
          'batch_size': 4,
          'lr':1e-5,
          'epochs': 2,
          'weight_decay': 0.01,
          'margin': 0.45,
          'num_bins':8,
          'repeats':6,
          'pct_pairs':0.4,
          'pct_pairs_split':0.4,
          'model_type':'regular',
          'pooling_type': 'mean_pooling',
          'only_top_layer': False,
          'do_early_stop': False,
          'early_stop_loss': 0.20,
          'start_batches':4000,
          'output_file_name': '/content/drive/MyDrive/commonlit/models/mid_kfold1/model2_fold',
          }

