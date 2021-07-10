import torch

from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoModel

device = 'cuda'


class CommonlitModel(PreTrainedModel):
    def __init__(self, module_config, config):
        super(CommonlitModel, self).__init__(module_config)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['pretrained_model_name_or_path'],
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
        )

        # Train only the top layer
        if config['only_top_layer']:
            # Need to change to general instead of only Roberta
            for param in self.model.roberta.parameters():
                param.requires_grad = False

        self.books_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['books_margin'])
        self.science_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['science_margin'])
        self.simple_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['simple_margin'])
        self.pairs_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['pairs_margin'])

        # torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, batch):

        diff = None

        # Calculate MSE loss for single text batch sample
        if batch['task_name'] in ['commonlit', 'commonlit_base', 'augmented_commonlit']:

            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_masks'].to(device)
            b_labels = batch['labels'].to(device)

            out = self.model(b_input_ids,
                             attention_mask=b_input_mask,
                             labels=b_labels)

            logits = out['logits']
            loss = out['loss']

        # Need logits for each text for pairwise comparison
        else:

            text1 = batch['text1']
            text2 = batch['text2']

            b_input_ids_1 = text1['input_ids'].to(device)
            b_input_mask_1 = text1['attention_masks'].to(device)
            b_labels_1 = text1['labels'].to(device)

            b_input_ids_2 = text2['input_ids'].to(device)
            b_input_mask_2 = text2['attention_masks'].to(device)
            b_labels_2 = text2['labels'].to(device)

            out1 = self.model(b_input_ids_1,
                              attention_mask=b_input_mask_1,
                              labels=b_labels_1.float())

            out2 = self.model(b_input_ids_2,
                              attention_mask=b_input_mask_2,
                              labels=b_labels_2.float())

            logits1 = out1['logits']
            logits2 = out2['logits']
            logits = out1['logits']  # For output

            # Loss is cross-entropy with implied probability from Commonlit target
            if batch['task_name'] in ['commonlit_pairs', 'commonlit_base_pairs']:

                loss = self.cross_entropy(logits1.view(-1, 1),
                                          logits2.view(-1, 1),
                                          b_labels_1,
                                          b_labels_2,
                                          )

            # Loss is ranking loss
            else:
                # Margin to prevent from collapse to mean

                if batch['task_name'] == 'commonlit_pairs_split':
                    # Convert logit target difference to -1/1 for correct Ranking Loss
                    loss = self.pairs_ranking_loss_fn(logits1.view(-1, 1),
                                                      logits2.view(-1, 1),
                                                      2 * (b_labels_1 > b_labels_2).float() - 1)

                if batch['task_name'] == 'simple':
                    loss = self.simple_ranking_loss_fn(logits1.view(-1, 1),
                                                       logits2.view(-1, 1),
                                                       b_labels_1)

                if batch['task_name'] == 'books':

                    loss = self.books_ranking_loss_fn(logits1.view(-1, 1),
                                                      logits2.view(-1, 1),
                                                      b_labels_1)

                elif batch['task_name'] == 'science':

                    loss = self.science_ranking_loss_fn(logits1.view(-1, 1),
                                                        logits2.view(-1, 1),
                                                        b_labels_1)

                diff = (logits1 > logits2).float().sum()

        out = {'loss': loss,
               'logits': logits,
               'task_name': batch['task_name'],
               'diff': diff}

        return out

    def cross_entropy(self, logits1, logits2, target1, target2):

        # Cross entropy between model and label pair probabilities
        # Works for implied probabilities and 0/1 labels
        log_q = torch.nn.functional.log_softmax(torch.stack([logits1, logits2], dim=1))
        p = torch.nn.functional.softmax(torch.stack([target1.float(), target2.float()], dim=1))
        cross_entropy = -(p * log_q).sum() / p.shape[0]

        return cross_entropy


class CommonlitCustomHeadModelInit(PreTrainedModel):
    def __init__(self, module_config, config):
        super(CommonlitCustomHeadModelInit, self).__init__(module_config)

        self.model = model = AutoModel.from_pretrained(
            config['pretrained_model_name_or_path'],
        )

        self.hidden_size = module_config.hidden_size
        self.pooling_type = config['pooling_type']

        # Train only the top layer
        if config['only_top_layer']:
            # Need to change to general instead of only Roberta
            for param in self.model.roberta.parameters():
                param.requires_grad = False

        self.head = RobertaHead(self.hidden_size, 1)
        self.mse_loss_fn = torch.nn.MSELoss()
        self.books_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['books_margin'])
        self.science_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['science_margin'])
        self.simple_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['simple_margin'])
        self.pairs_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=config['pairs_margin'])

        # torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def pool(self, out, attn_mask=None):

        if self.pooling_type == 'cls':

            out = out[0]
            out = out[:, 0, :]

        elif self.pooling_type == 'pooled':

            out = out[1]

        elif self.pooling_type == 'mean_pooling':

            out = self.mean_pooling(out, attn_mask)

        return out

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """

        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def forward(self, batch):

        diff = None

        # Calculate MSE loss for single text batch sample
        if batch['task_name'] == 'commonlit':

            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_masks'].to(device)
            b_labels = batch['labels'].to(device)

            out = self.model(b_input_ids,
                             attention_mask=b_input_mask,
                             )

            logits = self.head(self.pool(out, b_input_mask))
            loss = self.mse_loss_fn(logits, b_labels)

        # Need logits for each text for pairwise comparison
        else:

            text1 = batch['text1']
            text2 = batch['text2']

            b_input_ids_1 = text1['input_ids'].to(device)
            b_input_mask_1 = text1['attention_masks'].to(device)
            b_labels_1 = text1['labels'].to(device)

            b_input_ids_2 = text2['input_ids'].to(device)
            b_input_mask_2 = text2['attention_masks'].to(device)
            b_labels_2 = text2['labels'].to(device)

            out1 = self.model(b_input_ids_1,
                              attention_mask=b_input_mask_1,
                              )

            out2 = self.model(b_input_ids_2,
                              attention_mask=b_input_mask_2,
                              )

            logits1 = self.head(self.pool(out1, b_input_mask_1))
            logits2 = self.head(self.pool(out2, b_input_mask_2))

            logits = logits1  # For output

            # Loss is cross-entropy with implied probability from Commonlit target
            if batch['task_name'] == 'commonlit_pairs':

                loss = self.cross_entropy(logits1.view(-1, 1),
                                          logits2.view(-1, 1),
                                          b_labels_1,
                                          b_labels_2,
                                          )

            # Loss is ranking loss
            else:
                # Margin to prevent from collapse to mean

                if batch['task_name'] == 'commonlit_pairs_split':
                    # Convert logit target difference to -1/1 for correct Ranking Loss
                    loss = self.pairs_ranking_loss_fn(logits1.view(-1, 1),
                                                      logits2.view(-1, 1),
                                                      2 * (b_labels_1 > b_labels_2).float() - 1)

                if batch['task_name'] == 'simple':
                    loss = self.simple_ranking_loss_fn(logits1.view(-1, 1),
                                                       logits2.view(-1, 1),
                                                       b_labels_1)

                if batch['task_name'] == 'books':

                    loss = self.books_ranking_loss_fn(logits1.view(-1, 1),
                                                      logits2.view(-1, 1),
                                                      b_labels_1)

                elif batch['task_name'] == 'science':

                    loss = self.science_ranking_loss_fn(logits1.view(-1, 1),
                                                        logits2.view(-1, 1),
                                                        b_labels_1)

                diff = (logits1 > logits2 + self.margin).float().sum()

        out = {'loss': loss,
               'logits': logits,
               'task_name': batch['task_name'],
               'diff': diff}

        return out

    def cross_entropy(self, logits1, logits2, target1, target2):

        # Cross entropy between model and label pair probabilities
        # Works for implied probabilities and 0/1 labels
        log_q = torch.nn.functional.log_softmax(torch.stack([logits1, logits2], dim=1))
        p = torch.nn.functional.softmax(torch.stack([target1.float(), target2.float()], dim=1))
        cross_entropy = -(p * log_q).sum() / p.shape[0]

        return cross_entropy


class RobertaHead(torch.nn.Module):

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x