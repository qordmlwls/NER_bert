# !pip3 install pytorch-lightning-bolts seqeval seaborn scikit-plot
# !pip3 install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension
from tqdm import tqdm
import itertools
from functools import partial
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb #서버에서 설치 후 api등록해줘야 함
import itertools

import joblib

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
sns.set(color_codes=True)
sns.set(font_scale=1)
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")
# 텍스트 전처리
import re
import emoji
from soynlp.normalizer import repeat_normalize


def clean(x):
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x
class config:
    WANDB_NAME = 'BERT-tagger-final'
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    LOG_MODEL = True
    EPOCHS = 10
    TRAIN_VAL_RATIO = 0.001
    BASE_MODEL_PATH = "beomi/KcELECTRA-base"
    TOKENIZER = "beomi/KcELECTRA-base"
    # data_loader_cpu_workers = os.cpu_count()
    data_loader_cpu_workers = 1
    is_data_clean = False
    if is_data_clean:
        TRAINING_FILE = "../input/train_clean.csv"
        TEST_FILE = "../input/test_clean.csv"
    else:
        TRAINING_FILE = os.path.join('..', 'input', 'sns_ner_it_train.csv')
        TEST_FILE = os.path.join('..', 'input', 'sns_ner_it_test.csv')
        # TRAINING_FILE = "../input/sns_ner_it_train.csv"
        # TEST_FILE = "../input/sns_ner_it_test.csv"


        # TRAINING_FILE = "../input/Naver_NER_train2.csv"
        # TEST_FILE = "../input/Naver_NER_test2.csv"
        # TRAINING_FILE = "../input/train.csv"
        # TEST_FILE = "../input/test.csv"        
    TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        do_lower_case=False, use_fast=False
    )
def process_data(data_path, is_test=False, is_filter=True, min_length=4):
    df = pd.read_csv(data_path)
    if is_filter:
        df_filtered = df[df['Idx'] == 1]
    else:
        df_filtered = df

    sentences = df_filtered.groupby("SentId")["Word"].apply(list).values

    if not is_test:
        ## 특정 길이 이하는 제외하는듯
        sen_len = [len(s) for s in sentences]
        sorted_seq = sorted(sen_len)
        n_below_thresh = sorted_seq.index(min_length)
        s = sorted_seq[n_below_thresh]

        indices = np.argsort(np.array(sen_len))[n_below_thresh:]
        trunc_sen = sentences[indices]

    else:
        df_filtered['SentId'] = df_filtered['SentId'].apply(lambda x:x.split()[-1]).astype(int)
        sentences = df_filtered.groupby("SentId")["Word"].apply(list).values
        trunc_sen = sentences

    tag, trunc_tag, enc_tag = None, None, None
    if not is_test:
        enc_tag = preprocessing.LabelEncoder()
        df_filtered.loc[:, "Tag"] = enc_tag.fit_transform(df_filtered["Tag"])
        tag = df_filtered.groupby("SentId")["Tag"].apply(list).values
        trunc_tag = tag[indices]

    return trunc_sen, trunc_tag, enc_tag


class NERDataset:
    def __init__(self, texts, n_tags, tags=None, is_test=False):
        self.texts = texts
        self.tags = tags
        self.n_tags = n_tags
        self.is_test = is_test

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        text = [str(i) for i in text]
        if not self.is_test:
            tags = self.tags[item]
        ids = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                str(s),
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            if not self.is_test:
                target_tag.extend([tags[i]] * input_len)
            else:
                target_tag.extend([0] * input_len)

        ids = ids[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_tag = [self.n_tags - 1] + target_tag + [self.n_tags - 1]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        labels = target_tag + ([0] * padding_len)
        try:
            a = ' '.join(text)
        except:

            print(text)
            print([str(i) for i in text])
        return {
            "input_texts": ' '.join(text),
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }




class NERDataModule(pl.LightningDataModule):

    def __init__(self, train_texts, train_tags, val_texts, val_tags, test_texts, test_tags, train_batch, val_batch,
                 n_tags):
        super().__init__()
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.test_texts = test_texts
        self.train_tags = train_tags
        self.test_tags = test_tags
        self.val_tags = val_tags
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.n_tags = n_tags

    def setup(self):
        self.train_dataset = NERDataset(self.train_texts, self.n_tags, self.train_tags)
        self.val_dataset = NERDataset(self.val_texts, self.n_tags, self.val_tags)
        self.test_dataset = NERDataset(self.test_texts, self.n_tags, self.test_tags, is_test=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch,
            # shuffle=True,
            num_workers=config.data_loader_cpu_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch,
            num_workers=config.data_loader_cpu_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=64,
            num_workers=config.data_loader_cpu_workers
        )


def log_confusion_matrix(y_true=None, y_pred=None, labels=None, true_labels=None,
                         pred_labels=None, normalize=False):
    """
    Computes the confusion matrix to evaluate the accuracy of a classification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        true_label_indexes = np.in1d(classes, true_labels)
        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        pred_label_indexes = np.in1d(classes, pred_labels)
        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]

    data = []
    count = 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if labels is not None and (isinstance(pred_classes[i], int)
                                   or isinstance(pred_classes[0], np.integer)):
            pred_dict = labels[pred_classes[i]]
            true_dict = labels[true_classes[j]]
        else:
            pred_dict = pred_classes[i]
            true_dict = true_classes[j]
        data.append([pred_dict, true_dict, cm[i, j]])
        count += 1

    return data


class NERModel(pl.LightningModule):

    def __init__(self, n_tags: int, labels_list, steps_per_epoch=None, n_epochs=None, tokenizer=config.TOKENIZER):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_tags = n_tags
        self.labels_list = labels_list
        self.bert = AutoModel.from_pretrained(config.BASE_MODEL_PATH)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.n_tags)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs

    def mod_cross_entropy(self, attention_mask, logits, labels):
        lfn = nn.CrossEntropyLoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.n_tags)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(lfn.ignore_index).type_as(labels)
        )
        loss = lfn(active_logits, active_labels)
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        drop_out = self.dropout(bert_out[0])
        class_out = self.classifier(drop_out)
        loss = 0
        if labels != None:
            loss = self.mod_cross_entropy(attention_mask, class_out, labels)
        return loss, class_out

    def training_step(self, batch, batch_idx):
        input_texts = batch["input_texts"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch["token_type_ids"]
        loss, outputs = self(input_ids, attention_mask, token_type_ids, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "input_texts": input_texts, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_texts = batch["input_texts"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch["token_type_ids"]
        loss, outputs = self(input_ids, attention_mask, token_type_ids, labels)
        # self.log("val_loss", loss, prog_bar=True, logger=False)
        return {"loss": loss, "input_texts": input_texts, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_texts = batch["input_texts"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        loss, outputs = self(input_ids, attention_mask, token_type_ids)
        return {"input_texts": input_texts, "predictions": outputs}

    def test_epoch_end(self, outputs):
        tokenizer = self.tokenizer
        label_map = {i: label for i, label in enumerate(self.labels_list)}
        word_tokens, logits_list, preds_list, text_list = [], [], [], []

        for output in outputs:
            logits_net = output["predictions"].detach().cpu().numpy()
            for i in range(len(output["input_texts"])):
                sentence = output["input_texts"][i]
                text_list.append(sentence)

                tokens = []
                for s in (sentence).split():
                    t = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(str(s))))
                    tokens.extend(t[1:-1])

                if len(tokens) >= config.MAX_LEN:
                    print('-----')
                    print(sentence)
                    print('-----')
                    tokens = tokens[0:config.MAX_LEN - 2]

                word_tokens.append(tokens)
                preds = torch.argmax(output["predictions"], dim=2)[i, 1:len(tokens) + 1].tolist()
                logits = logits_net[i, 1:len(tokens) + 1]

                word_logits = []
                tmp = []
                for n, lab in enumerate(tokens):
                    if not (lab.startswith('##')):
                        if n != 0:
                            word_logits.append(tmp)
                        tmp = [list(logits[n])]
                    else:
                        tmp.append(list(logits[n]))
                word_logits.append(tmp)

                logits_list.append(word_logits)
                preds_list.append(preds)

        out_preds_list = [[] for _ in range(len(preds_list))]

        for i in range(len(preds_list)):
            for j in range(len(preds_list[i])):
                if not (word_tokens[i][j].startswith('##')):
                    try:
                      out_preds_list[i].append(label_map[preds_list[i][j]])
                    except:
                      pass

        predictions = [
            [{word: out_preds_list[i][j]} for j, word in enumerate(sentence.split()[: len(out_preds_list[i])])]
            for i, sentence in enumerate(text_list)
        ]

        joblib.dump(predictions, 'predictions.bin')
        return predictions

    def validation_epoch_end(self, outputs):
        tokenizer = self.tokenizer
        label_map = {i: label for i, label in enumerate(self.labels_list)}
        word_tokens, logits_list, preds_list, labels_list, text_list = [], [], [], [], []
        av_loss = []

        for output in outputs:
            logits_net = output["predictions"].detach().cpu().numpy()
            for i in range(len(output["input_texts"])):
                sentence = output["input_texts"][i]
                text_list.append(sentence)

                tokens = []
                for s in (sentence).split():
                    t = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(str(s))))
                    tokens.extend(t[1:-1])

                if len(tokens) >= config.MAX_LEN:
                    print('-----')
                    print(sentence)
                    print('-----')
                    tokens = tokens[0:config.MAX_LEN - 2]

                word_tokens.append(tokens)
                preds = torch.argmax(output["predictions"], dim=2)[i, 1:len(tokens) + 1].tolist()
                logits = logits_net[i, 1:len(tokens) + 1]

                labels = output["labels"][i, 1:len(tokens) + 1].tolist()

                word_logits = []
                tmp = []
                for n, lab in enumerate(tokens):
                    if not (lab.startswith('##')):
                        if n != 0:
                            word_logits.append(tmp)
                        tmp = [list(logits[n])]
                    else:
                        tmp.append(list(logits[n]))
                word_logits.append(tmp)

                labels_list.append(labels)
                logits_list.append(word_logits)
                preds_list.append(preds)

            av_loss.append(output["loss"].mean().flatten().tolist()[0])

        out_labels_list = [[] for _ in range(len(labels_list))]
        out_preds_list = [[] for _ in range(len(preds_list))]

        for i in range(len(preds_list)):
            for j in range(len(preds_list[i])):
                if not (word_tokens[i][j].startswith('##')):
                  try:
                    out_labels_list[i].append(label_map[labels_list[i][j]])
                    out_preds_list[i].append(label_map[preds_list[i][j]])
                  except:
                    pass
        model_outputs = [
            [{word: logits_list[i][j]} for j, word in enumerate(sentence.split()[: len(out_preds_list[i])])]
            for i, sentence in enumerate(text_list)
        ]

        result = {
            # "eval_loss": np.mean(np.asarray(av_loss)),
            # "precision": precision_score(out_labels_list, out_preds_list),
            # "recall": recall_score(out_labels_list, out_preds_list),
            # "f1_score": f1_score(out_labels_list, out_preds_list)
        }

        report = classification_report(out_labels_list, out_preds_list)
        liststr = report.split('\n')
        listtag = liststr[3:-5]
        listavg = liststr[-4:-1]
        item_names, precision, recall, f1, count = [], [], [], [], []
        for val in listtag:
            items = val.split()
            item_names.append(items[0])
            precision.append(items[1])
            recall.append(items[2])
            f1.append(items[3])
            count.append(items[4])

        for val in listavg:
            items = val.split()
            item_names.append('#' + items[0] + '_' + items[1])
            precision.append(items[2])
            recall.append(items[3])
            f1.append(items[4])
            count.append(items[5])

        metrics = [item_names, precision, recall, f1, count]

        self.logger.experiment.log({"Metrics": wandb.Table(
            columns=['Name', 'Precision', 'Recall', 'F1', 'Count'],
            data=list(map(list, zip(*metrics))))})
        # print(out_preds_list[0],word_tokens[0],out_labels_list[0], sep='\n')
        # print(len(preds_list[5]),len(word_tokens[5]),len(labels_list[5]))
        # self.log("eval_loss", result["eval_loss"], logger=True)
        # self.log("precision", result["precision"], logger=True)
        # self.log("recall", result["recall"], logger=True)
        # self.log("f1_score", result["f1_score"], prog_bar=True, logger=True)
        # self.log("f1_score", result["f1_score"], logger=True)
        truth = [tag for out in out_labels_list for tag in out]
        preds = [tag for pred_out in out_preds_list for tag in pred_out]

        truth_trun = []
        preds_trun = []

        for i, out in enumerate(out_labels_list):
            pred_out = out_preds_list[i]
            for j, true in enumerate(out):
                pred = pred_out[j]
                if not (pred == true and pred == 'O'):
                    truth_trun.append(true)
                    preds_trun.append(pred)

        model_outputs_less = [[logits_list[i][j] for j in range(len(out_preds_list[i]))] for i in
                              range(len(out_preds_list))]

        outputs = [np.mean(logits, axis=0) for output in model_outputs_less for logits in output]

        # ROC
        self.logger.experiment.log({"ROC": wandb.plots.ROC(truth, outputs, labels_list)})

        # Precision Recall
        self.logger.experiment.log({"PR": wandb.plots.precision_recall(truth, outputs, labels_list)})
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(truth_trun, preds_trun, ax=ax)

        self.logger.experiment.log({f'confusion_matrix_img: epoch {self.trainer.current_epoch}': wandb.Image(fig)})

        data = log_confusion_matrix(truth, preds, self.labels_list)
        self.logger.experiment.log({"confusion_matrix": wandb.Table(
            columns=['Predicted', 'Actual', 'Count'],
            data=data)})

    def configure_optimizers(self):

        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(
                        nd in n for nd in no_decay
                    )
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(
                        nd in n for nd in no_decay
                    )
                ],
                "weight_decay": 0.0,
            },
        ]

        num_warmup_steps = 0
        num_train_steps = self.steps_per_epoch * self.n_epochs - num_warmup_steps
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )

        return [optimizer], [scheduler]

class LoggingCallback(Callback):
    def on_init_end(self, trainer):
        trainer.logger.experiment.log({'model_name': config.BASE_MODEL_PATH, 'is_data_clean': config.is_data_clean,
                                 'epochs': config.EPOCHS,  'batch_size' : config.TRAIN_BATCH_SIZE, 'train_val_ratio' : config.TRAIN_VAL_RATIO, 'token_max': config.MAX_LEN})



class NerTrainer:
    def __init__(self):
        self.num_tag = None
        self.enc_tag = None
        self.train_sentences = None
        self.data_module = None
    def preprocess_data(self):
        sentences, tag, self.enc_tag = process_data(config.TRAINING_FILE, is_filter=False)
        test_sentences, test_tag, _ = process_data(config.TEST_FILE, is_filter=False, is_test=True)
        meta_data = {
            "enc_tag": self.enc_tag
        }
        joblib.dump(meta_data, "meta.bin")
        self.num_tag = len(list(self.enc_tag.classes_))
        self.train_sentences, val_sentences, train_tag, val_tag = train_test_split(
            sentences,
            tag,
            random_state=42,
            test_size=config.TRAIN_VAL_RATIO
        )
        self.data_module = NERDataModule(self.train_sentences, train_tag, val_sentences, val_tag, test_sentences, test_tag,
                                    config.TRAIN_BATCH_SIZE, config.VALID_BATCH_SIZE, self.num_tag)
        self.data_module.setup()
    def train_model(self):
        wandb_logger = WandbLogger(name=config.WANDB_NAME, project='Named Entity Recognition',
                                   log_model=config.LOG_MODEL, save_code=True)
        label_list = self.enc_tag.inverse_transform(range(self.num_tag)).tolist()
        model = NERModel(self.num_tag, label_list, steps_per_epoch=len(self.train_sentences) // config.TRAIN_BATCH_SIZE,
                         n_epochs=config.EPOCHS)
        # Instantiate LearningRateMonitor Callback
        # lr_logger = LearningRateMonitor(logging_interval='epoch')

        # wandb_logger.experiment.log({'model_name': config.BASE_MODEL_PATH, 'is_data_clean': config.is_data_clean,
        #  'epochs': config.EPOCHS,  'batch_size' : config.TRAIN_BATCH_SIZE, 'train_val_ratio' : config.TRAIN_VAL_RATIO, 'token_max': config.MAX_LEN})
        trainer = pl.Trainer(weights_summary='full', max_epochs=config.EPOCHS, deterministic=torch.cuda.is_available(),
                             gpus=-1 if torch.cuda.is_available() else None,

                             accelerator='dp', progress_bar_refresh_rate=10,
                             callbacks=[PrintTableMetricsCallback(), LoggingCallback()], logger=wandb_logger)

        trainer.fit(model, self.data_module)
    def run(self):
        self.preprocess_data()
        self.train_model()

