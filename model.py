import os,random,math,codecs,logging,shutil
from transformers import TFBertForSequenceClassification,create_optimizer,BertTokenizer,BertConfig
import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score,classification_report


logging.disable(30)
logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        # filename='log.txt',
                        # filemode='w'
                    )
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class BaseArguments(object):
    def __init__(self):
        self.model_name_or_path = ["bert-base-chinese"][0]
        self.tokenizer_dir = ["bert-base-chinese"][0]
        self.output_dir = "data"
        self.data_dir = "data/original_data"
        self.seed = 42
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.strategy,self.gpus_nums = self.get_strategy()

        self.batch_size = 10
        self.learning_rate = 1e-5
        self.num_train_epochs = 4
        self.logging_steps = 1
        self.save_total_limit = 1

    def get_strategy(self):
        gpus = tf.config.list_physical_devices("GPU")
        gpus_nums = len(gpus)

        if gpus_nums == 0:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        elif gpus_nums == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        elif gpus_nums > 1:
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            strategy = tf.distribute.MirroredStrategy()
        else:
            raise ValueError("Cannot find the proper strategy please check your environment properties.")

        return strategy,gpus_nums


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_metrics(y_true, y_pred) -> float:
    f1 = f1_score(y_true, y_pred, average='macro')
    # report = classification_report(y_true,y_pred,output_dict=True)
    # print(report)
    return f1


class TFTrainer:
    def __init__(self,):
        self.args = BaseArguments()
        self.label2id = {line.split("\t")[0]:int(line.split("\t")[1]) for line in open(os.path.join(self.args.data_dir,"label.txt"),encoding="utf-8",mode="r").read().splitlines() if len(line.split("\t"))==2}
        config = BertConfig.from_pretrained(self.args.model_name_or_path,num_labels=len(self.label2id))
        self.model = TFBertForSequenceClassification(config=config)
        self.compute_metrics = compute_metrics
        # self.tb_writer = tf.summary.create_file_writer(os.path.join(self.args.output_dir,"log"))
        print(self.label2id)
        set_seed(self.args.seed)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.tokenizer_dir)
        self.train_dataset = self.read_dataset("train.txt")
        self.eval_dataset = self.read_dataset("eval.txt")[:]

    def train_generator(self,content):
        random.shuffle(content)
        for line in content:
            if type(line) == bytes:
                line = line.decode()
            line_ele = line.strip().split("\t")
            if len(line_ele) == 3:
                input_ids = self.tokenizer.encode_plus(line_ele[1], add_special_tokens=True, return_token_type_ids=True,
                                           return_attention_mask=True,
                                           max_length=self.model.config.max_position_embeddings)["input_ids"]
                # print(input_ids)
                yield input_ids, [self.label2id[line_ele[-1]]]

    def get_dataloader(self):
        dataset_fn = lambda x:tf.data.Dataset.from_generator(self.train_generator, output_types=(tf.int32, tf.int32),args=[x]).padded_batch(self.args.batch_size, padded_shapes=([None], [None]), padding_values=(self.tokenizer.pad_token_id, 1))

        train_dataloader = dataset_fn(self.train_dataset)
        eval_dataloader = dataset_fn(self.eval_dataset)

        return train_dataloader,eval_dataloader

    def read_dataset(self,txt_file):
        content = codecs.open(os.path.join(self.args.data_dir,txt_file), encoding="utf-8", mode="r").read().splitlines()
        random.shuffle(content)
        return content

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer, lr_scheduler = create_optimizer(self.args.learning_rate,num_training_steps,self.args.warmup_steps,adam_epsilon=self.args.adam_epsilon,weight_decay_rate=self.args.weight_decay)
        return optimizer, lr_scheduler

    def train(self) -> None:
        train_example_nums,eval_example_nums = len(self.train_dataset),len(self.eval_dataset)
        num_steps = math.ceil(train_example_nums / self.args.batch_size)
        t_total = num_steps * self.args.num_train_epochs
        epochs = self.args.num_train_epochs

        with self.args.strategy.scope():
            self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler(num_training_steps=t_total)

            logger.error("***** Running training *****")
            logger.error(f"  Train Num examples per epoch = {train_example_nums}")
            logger.error(f"  Eval Num examples per epoch = {eval_example_nums}")
            logger.error(f"  Num Epochs = {epochs}")
            logger.error(f"  Instantaneous batch size = {self.args.batch_size}")
            logger.error(f"  Train Steps per epoch = {num_steps}")
            logger.error(f"  Eval Steps per epoch = {math.ceil(eval_example_nums / self.args.batch_size)}")
            logger.error(f"  Total optimization steps = {t_total}")

            global_step = 0
            best_f1 = 0
            self.train_loss = tf.keras.metrics.Sum()
            for epoch_curr in range(int(epochs)):
                train_dataloader,eval_dataloader = self.get_dataloader()
                for step, batch in enumerate(train_dataloader):
                    global_step += 1
                    self.distributed_training_steps(batch)
                    training_loss = self.train_loss.result()
                    self.train_loss.reset_states()

                    if global_step % self.args.logging_steps == 0:
                        preds, label_ids, f1 = self.evaluate(eval_dataloader)
                        step_loss_lr_f1 = f"step:{global_step}-loss{training_loss.numpy():.4f}:-lr:{self.lr_scheduler(global_step).numpy()}-f1:{f1:.6f}"
                        logger.error(step_loss_lr_f1)
                        if f1 > best_f1:
                            score_up = f"F1 from {best_f1} to {f1}"
                            logger.error(score_up)
                            best_f1 = f1
                            self.save_model(epoch_curr,global_step,best_f1)

    def training_step(self, features, labels, nb_instances_in_global_batch):
        per_example_loss, _ = self.run_model(features, labels, True)
        scaled_loss = per_example_loss / tf.cast(nb_instances_in_global_batch, dtype=per_example_loss.dtype)
        self.train_loss.update_state(per_example_loss)
        gradients = tf.gradients(scaled_loss, self.model.trainable_variables)
        gradients = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)]
        self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
        return gradients

    @tf.function
    def distributed_training_steps(self, batch):
        with self.args.strategy.scope():
            inputs = self._get_step_inputs(batch)
            self.args.strategy.run(self.training_step, inputs)

    @staticmethod
    def _get_step_inputs(batch):
        features, labels = batch
        if isinstance(labels, PerReplica):
            labels = tf.concat(labels.values, axis=0)
        nb_instances = tf.reduce_sum(tf.cast(labels != -100, dtype=tf.int32))

        if isinstance(labels, PerReplica):
            # need to make a `PerReplica` objects for ``nb_instances``
            nb_instances = PerReplica([nb_instances] * len(labels.values))
        step_inputs = (features, labels, nb_instances)
        return step_inputs

    def run_model(self, features, labels, training):
        outputs = self.model(input_ids=features,labels=labels,training=training)[:2]
        loss, logits = outputs[:2]
        return loss, logits

    def evaluate(self, dataset: tf.data.Dataset = None):
        label_ids: np.ndarray = None
        preds: np.ndarray = None

        for step, batch in enumerate(dataset):
            logits = self.distributed_prediction_steps(batch)
            _, labels = batch

            if isinstance(logits, tuple):
                logits = logits[0]

            if isinstance(labels, tuple):
                labels = labels[0]

            if self.args.strategy.num_replicas_in_sync > 1:
                for val in logits.values:
                    if preds is None:
                        preds = val.numpy()
                    else:
                        preds = np.append(preds, val.numpy(), axis=0)

                for val in labels.values:
                    if label_ids is None:
                        label_ids = val.numpy()
                    else:
                        label_ids = np.append(label_ids, val.numpy(), axis=0)
            else:
                if preds is None:
                    preds = logits.numpy()
                else:
                    preds = np.append(preds, logits.numpy(), axis=0)

                if label_ids is None:
                    label_ids = labels.numpy()
                else:
                    label_ids = np.append(label_ids, labels.numpy(), axis=0)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            label_ids = label_ids.flatten()
            preds = np.argmax(preds, axis=1)
            metrics = self.compute_metrics(label_ids, preds)
        else:
            metrics = {}
        return preds, label_ids, metrics

    def prediction_step(self, features: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        per_example_loss, logits = self.run_model(features, labels, False)
        return logits

    @tf.function
    def distributed_prediction_steps(self, batch):
        logits = self.args.strategy.run(self.prediction_step, batch)
        return logits

    def save_model(self, epoch, global_step, best_score):
        file_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}-step-{global_step}-f1-{best_score:.6f}")
        os.makedirs(file_dir, exist_ok=True)
        logger.error("Saving model checkpoint to %s", file_dir)
        self.model.save_pretrained(file_dir)
        self._rotate_checkpoints(self.args.output_dir)

    def _rotate_checkpoints(self, output_dir) -> None:
        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.error("Deleting older checkpoint [{}] due to args.save_total_limit:{}".format(checkpoint,self.args.save_total_limit))
            shutil.rmtree(checkpoint)

    def _sorted_checkpoints(self, output_dir):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"checkpoint-*")]

        for path in glob_checkpoints:
            try:
                tup = (float(path.split("-")[-1]), path)
                ordering_and_checkpoint_path.append(tup)
            except:
                logger.info("")
        checkpoints_sorted = sorted(ordering_and_checkpoint_path, key=lambda x: x[0])
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted


if __name__ == '__main__':
    trainer = TFTrainer()
    trainer.train()


