import numpy as np
import pandas as pd

import torch
from transformers import AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm

from .utils import add_center_to_folds
from .stats import TrainingStats
from .dataloaders import generate_dataloaders
from .model import CommonlitModel, CommonlitCustomHeadModelInit

ROOT_PATH = '/content/drive/MyDrive/commonlit'
DATA_PATH = ROOT_PATH + '/data'
FOLD_PATH = DATA_PATH + '/los_folds.csv'

device = 'cuda'


def run(fold, model_num, config):
    # Load CommonLit train dataset
    train = pd.read_csv(FOLD_PATH)
    # Add the sample with target 0 to all the folds
    train = add_center_to_folds(train)
    # Texts
    train_x = train.excerpt
    # Targets
    train_y = train.target

    # Split data in train and valid according to fold
    train_idx = train[train.kfold != fold].index
    valid_idx = train[train.kfold == fold].index

    # Keep track of best model so far
    curr_best = config['curr_best']
    curr_best_calibrated = 1e15

    # Statistics gatherer
    stats = TrainingStats()

    # Generate data loaders for train and valid
    train_dataloader, valid_dataloader = generate_dataloaders(train_x,
                                                              train_y,
                                                              train_idx,
                                                              valid_idx,
                                                              batch_size=config['batch_size'],
                                                              tokenizer_name=config['tokenizer_name'],
                                                              num_bins=config['num_bins'],
                                                              repeats=config['repeats'],
                                                              pct_pairs=config['pct_pairs'],
                                                              pct_simple=config['pct_simple'],
                                                              pct_pairs_split=config['pct_pairs_split'],
                                                              frac_science=config['frac_science'],
                                                              frac_books=config['frac_books'],
                                                              num_base_samples=config['num_base_samples'],
                                                              augment=config['augment'])

    # Load model, num_labels=1 specifies a regression with MSE loss
    module_config = AutoConfig.from_pretrained(config['pretrained_model_name_or_path'])
    if config['model_type'] == 'regular':
        model = CommonlitModel(module_config, config).to(device)
    elif config['model_type'] == 'custom':
        model = CommonlitCustomHeadModelInit(module_config, config).to(device)

    optimizer = AdamW(model.parameters(),
                      lr=config['lr'],
                      weight_decay=config['weight_decay'])

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * config['epochs']

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    early_stop = False

    for epoch_i in range(0, config['epochs']):

        if config['do_early_stop']:
            if early_stop:
                break

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config['epochs']))

        for count_batches, batch in enumerate(tqdm(train_dataloader)):

            # ========================================
            #               Validation
            # ========================================

            if ((count_batches + 1) % config['partial_eval'] == 0) and (
                    (count_batches >= config['start_batches']) or epoch_i >= 1):

                model.eval()

                stats.total_eval_loss = 0

                logits_calibrate = []
                y_true_calibrate = []

                for batch_valid in valid_dataloader:
                    # Avoid gradient updates
                    with torch.no_grad():
                        out = model(batch_valid)
                        loss = out['loss']
                        y_true_calibrate.append(batch_valid['labels'].cpu().numpy())
                        logits_calibrate.append(out['logits'].cpu().numpy())

                    stats.update_eval(loss.item(),
                                      config['batch_size'] * count_batches,
                                      len(valid_dataloader),
                                      optimizer.param_groups[0]["lr"],
                                      batch['task_name'])

                logits_calibrate = np.concatenate(logits_calibrate, 0)
                y_true_calibrate = np.concatenate(y_true_calibrate, 0)
                mid_point = logits_calibrate[np.where(y_true_calibrate == 0)][0]
                mu, sigma = norm.fit(logits_calibrate - y_true_calibrate)

                rmse_calibrated = mean_squared_error(logits_calibrate - mu, y_true_calibrate, squared=False)
                stats.rmse_calibrated = rmse_calibrated
                stats.mid_point = mid_point
                stats.mu = mu

                if config['do_early_stop']:
                    if stats.avg_commonlit_train_loss < config['early_stop_loss']:
                        early_stop = True
                        break

                stats.print_valid()

                if stats.avg_val_loss < curr_best:
                    curr_best = stats.avg_val_loss
                    print('\n**************************')
                    print('Fold: ', fold, ' | rmse: ', np.sqrt(curr_best),
                          ' | rmse calib: ', rmse_calibrated, '| mse: ', curr_best, ' | mu: ', mu, ' | midpoint: ',
                          mid_point, '| task: ', batch['task_name'])
                    print('**************************\n')
                    model.model.save_pretrained(
                        config['output_file_name'] + 'fold' + str(fold) + 'model' + str(model_num))

            # ========================================
            #               Training
            # ========================================

            model.train()
            model.zero_grad()
            out = model(batch)
            loss = out['loss']
            diff = out['diff']
            stats.update_train(loss.item(), diff, batch['task_name'], config['batch_size'])

            # Backprop
            loss.backward()
            # Clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()


