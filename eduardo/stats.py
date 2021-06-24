import numpy as np
from collections import deque


class TrainingStats:

    def __init__(self):

        self.last_lr = 0

        self.batch_size = 4
        self.total_train_loss = 0
        self.total_eval_loss = 0
        self.total_simple_loss = 0
        self.total_books_loss = 0
        self.total_split_loss = 0
        self.total_commonlit_train_loss = 0
        self.total_split_correct = 0
        self.total_simple_correct = 0
        self.total_books_correct = 0

        self.count_batches = 0
        self.count_common_lit = 0
        self.count_split = 0
        self.count_simple = 0
        self.count_books = 0

        self.avg_train_loss = 0
        self.avg_commonlit_train_loss = 0
        self.avg_books_loss = 0
        self.avg_simple_loss = 0

        self.commonlit_train_loss_deque = deque(maxlen=100)
        self.avg_commonlit_train_loss_deque = 0

    def update_eval(self, loss, n_train, n_valid, lr, task_name):

        self.last_lr = lr
        self.task_name = task_name

        # Accumulate the validation loss.
        self.total_eval_loss += loss

        # Calculate the average loss over all of the batches.
        self.avg_val_loss = self.total_eval_loss / n_valid

        # Measure how long the validation run took.
        self.avg_train_loss = self.total_train_loss / (n_train)
        self.avg_commonlit_train_loss = self.total_commonlit_train_loss / (self.count_common_lit + 1e-15)
        self.avg_books_loss = self.total_books_loss / (self.count_books + 1e-15)
        self.avg_simple_loss = self.total_simple_loss / (self.count_simple + 1e-15)
        self.split_accuracy = self.total_split_correct / (self.count_split + 1e-15)
        self.books_accuracy = self.total_books_correct / (self.count_books + 1e-15)
        self.simple_accuracy = self.total_simple_correct / (self.count_simple + 1e-15)
        self.avg_commonlit_train_loss_deque = sum(self.commonlit_train_loss_deque) / (
                    self.batch_size * len(self.commonlit_train_loss_deque) + 1e-15)

    def update_train(self, loss, diff, task_name, batch_size):

        self.batch_size = batch_size

        # Calculate MSE loss for single text batch sample
        if task_name == 'commonlit':
            self.count_common_lit += batch_size
            self.total_commonlit_train_loss += loss
            self.commonlit_train_loss_deque.append(loss)

        # Need logits for each text for pairwise comparison
        else:

            # Loss is cross-entropy with implied probability from Commonlit target
            if task_name == 'commonlit_pairs':

                pass
            # Loss is ranking loss
            else:

                if task_name == 'commonlit_pairs_split':
                    self.count_split += batch_size
                    self.total_split_correct += diff
                    self.total_split_loss += loss

                if task_name == 'simple':

                    self.count_simple += batch_size
                    self.total_simple_correct += diff
                    self.total_simple_loss += loss

                elif task_name == 'books':

                    self.count_books += batch_size
                    self.total_books_correct += diff
                    self.total_books_loss += loss

        # Add losses from all loss functions
        self.total_train_loss += loss

    def print_valid(self):

        # stats_string = " Train: {:.4f} | Train Latest {:.4f} | Valid: {:.4f} | Split: {:.4f} | Books: {:.4f} | Simple: {:.4f} | lr: {:.4f}"
        stats_string = " Train: {:.4f} | Train Latest {:.4f} | Valid: {:.4f} | Books: {:.4f} | Simple: {:.4f} | lr: {:.4f}"

        print(stats_string.format(
            np.sqrt(self.avg_commonlit_train_loss),
            np.sqrt(self.avg_commonlit_train_loss_deque),
            np.sqrt(self.avg_val_loss),
            # self.split_accuracy,
            self.books_accuracy,
            self.simple_accuracy,
            10000 * self.last_lr),
            '| task name: {}'.format(self.task_name)
        )