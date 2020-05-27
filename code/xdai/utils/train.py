import logging, os, re, shutil, torch
from tqdm import tqdm
from typing import List
from xdai.utils.common import move_to_gpu
from xdai.utils.nn import enable_gradient_clipping, rescale_gradients


logger = logging.getLogger(__name__)


'''Update date: 2019-Nov-6'''
class MetricTracker:
    def __init__(self, should_decrease, patience=None):
        self._best_so_far = None
        self._patience = patience
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self.best_epoch_metrics = {}
        self._epoch_number = 0
        self.best_epoch = None
        self._should_decrease = should_decrease


    def clear(self) -> None:
        self._best_so_far = None
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self._epoch_number = 0
        self.best_epoch = None


    def state_dict(self):
        return {
            "best_so_far": self._best_so_far,
            "patience": self._patience,
            "epochs_with_no_improvement": self._epochs_with_no_improvement,
            "is_best_so_far": self._is_best_so_far,
            "should_decrease": self._should_decrease,
            "best_epoch_metrics": self.best_epoch_metrics,
            "epoch_number": self._epoch_number,
            "best_epoch": self.best_epoch,
        }


    def load_state_dict(self, state_dict) -> None:
        self._best_so_far = state_dict["best_so_far"]
        self._patience = state_dict["patience"]
        self._epochs_with_no_improvement = state_dict["epochs_with_no_improvement"]
        self._is_best_so_far = state_dict["is_best_so_far"]
        self._should_decrease = state_dict["should_decrease"]
        self.best_epoch_metrics = state_dict["best_epoch_metrics"]
        self._epoch_number = state_dict["epoch_number"]
        self.best_epoch = state_dict["best_epoch"]


    def add_metric(self, metric):
        if self._best_so_far is None:
            new_best = True
        else:
            if self._should_decrease:
                if metric < self._best_so_far:
                    new_best = True
            else:
                if metric > self._best_so_far:
                    new_best = True

        if new_best:
            self.best_epoch = self._epoch_number
            self._is_best_so_far = True
            self._best_so_far = metric
            self._epochs_with_no_improvement = 0
        else:
            self._is_best_so_far = False
            self._epochs_with_no_improvement += 1
        self._epoch_number += 1


    def is_best_so_far(self) -> bool:
        return self._is_best_so_far


    def should_stop_early(self) -> bool:
        if self._patience is None:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py (batch_loss)
Update date: 2019-March-03'''
def _batch_loss(args, model, batch):
    batch = move_to_gpu(batch, cuda_device=args.cuda_device[0])
    output_dict = model(**batch)
    loss = output_dict.get("loss")
    return loss


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py (_validation_loss)
Update date: 2019-April-20'''
def _get_val_loss(args, model, iterator, data):
    model.eval()
    generator = iterator(data, shuffle=False)
    total_loss, batch_counter = 0.0, 0
    for batch in generator:
        batch_counter += 1
        _loss = _batch_loss(args, model, batch)
        if isinstance(_loss, float):
            total_loss += _loss
        else:
            total_loss += _loss.item()
    loss = float(total_loss / batch_counter) if batch_counter > 0 else 0.0
    return loss


'''Update date: 2019-April-20'''
def _is_best_model_so_far(this_epoch_score: float, score_per_epoch: List[float]):
    if not score_per_epoch:
        return True
    else:
        return this_epoch_score > max(score_per_epoch)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py
Update date: 2019-April-20'''
def _output_metrics_to_console(train_metrics, dev_metrics={}):
    metric_names = list(train_metrics.keys()) + list(dev_metrics.keys())
    metric_names = list(set(metric_names))
    train_metrics = ["%s: %s" % (k, str(train_metrics.get(k, 0))) for k in metric_names]
    logger.info(" # Train set \n     %s" % ("; ".join(train_metrics)))
    dev_metrics = ["%s: %s" % (k, str(dev_metrics.get(k, 0))) for k in metric_names]
    logger.info(" # Dev set \n     %s" % ("; ".join(dev_metrics)))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py#_save_checkpoint
Update date: 2019-Nov-9'''
def _save_checkpoint(model_dir, model, epoch, is_best=False):
    model_path = os.path.join(model_dir, "epoch_%s.th" % epoch)
    torch.save(model.state_dict(), model_path)
    if is_best:
        logger.info(" # Best dev performance so far. Copying weights to %s/best.th" % model_dir)
        shutil.copyfile(model_path, os.path.join(model_dir, "best.th"))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py
Update date: 2019-April-20'''
def _should_early_stop(score_per_epoch: List[float], patience=0):
    if patience > 0 and patience < len(score_per_epoch):
        return max(score_per_epoch[-patience:]) <= max(score_per_epoch[:-patience])
    return False


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py#_train_epoch
Update date: 2019-Nov-9'''
def _train_epoch(args, model, optimizer, iterator, data, shuffle=True):
    model.train()
    total_loss = 0.0
    generator = iterator(data, shuffle=shuffle)
    num_batches = iterator.get_num_batches(data)
    batch_counter = 0

    for batch in generator:
        batch_counter += 1
        optimizer.zero_grad()
        loss = _batch_loss(args, model, batch)
        loss.backward()
        total_loss += loss.item()
        rescale_gradients(model, args.max_grad_norm)
        optimizer.step()

        metrics = model.get_metrics(reset=False)
        metrics["loss"] = float(total_loss / batch_counter) if batch_counter > 0 else 0.0

        if batch_counter % args.logging_steps == 0 or batch_counter == num_batches:
            logger.info("%d out of %d batches, loss: %.3f" % (batch_counter, num_batches, metrics["loss"]))

    metrics = model.get_metrics(reset=True)
    metrics["loss"] = float(total_loss / batch_counter) if batch_counter > 0 else 0.0
    return metrics


'''Update date: 2019-Nov-9'''
def _check_max_save_checkpoints(output_dir, max_save_checkpoints, pattern=("epoch_", ".th")):
    if max_save_checkpoints < 0: return None
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith(pattern[0]) and f.endswith(pattern[1])]
    if len(checkpoints) > max_save_checkpoints:
        numbers = sorted([int(re.findall("\d+", filename)[0]) for filename in checkpoints], reverse=True)
        for n in numbers[max_save_checkpoints:]:
            os.remove(os.path.join(output_dir, "%s%d%s" % (pattern[0], n, pattern[1])))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py#train
Update date: 2019-Nov-9'''
def train_op(args, model, optimizer, train_data, train_iterator, dev_data, dev_iterator):
    enable_gradient_clipping(model, args.grad_clipping)
    model_dir = args.output_dir
    max_epoches = args.num_train_epochs
    patience = args.patience
    validation_metric = args.eval_metric

    validation_metric_per_epoch = []
    metrics = {}

    for epoch in range(0, max_epoches):
        logger.info("Epoch %d/%d" % (epoch + 1, max_epoches))
        train_metrics = _train_epoch(args, model, optimizer, train_iterator, train_data)
        with torch.no_grad():
            val_loss = _get_val_loss(args, model, dev_iterator, dev_data)
            val_metrics = model.get_metrics(reset=True)
            val_metrics["loss"] = val_loss
            this_epoch_val_metric = val_metrics[validation_metric]
            is_best = _is_best_model_so_far(this_epoch_val_metric, validation_metric_per_epoch)
            validation_metric_per_epoch.append(this_epoch_val_metric)

        _output_metrics_to_console(train_metrics, val_metrics)

        metrics["epoch"] = epoch
        for k, v in train_metrics.items():
            metrics["training_" + k] = v
        for k, v in val_metrics.items():
            metrics["validation_" + k] = v

        if is_best:
            metrics["best_epoch"] = epoch
            for k, v in val_metrics.items():
                metrics["best_validation_" + k] = v

        _save_checkpoint(model_dir, model, epoch, is_best)
        _check_max_save_checkpoints(args.output_dir, args.max_save_checkpoints)

        if _should_early_stop(validation_metric_per_epoch, patience):
            logger.info(" # Ran out of patience. Stopping training.")
            break
    return metrics


'''Update date: 2019-Nov-7'''
def eval_op(args, model, data, data_iterator):
    sentences, predictions = [], []
    with torch.no_grad():
        model.eval()
        generator = data_iterator(data, shuffle=False)
        total_loss = 0.0
        for batch in tqdm(generator, desc="Evaluating"):
            sentences += batch["sentence"]
            batch = move_to_gpu(batch, args.cuda_device[0])
            output_dict = model(**batch)
            loss = output_dict.get("loss", None)
            predictions += output_dict.get("preds")
            if loss:
                total_loss += loss.item()
        final_metrics = model.get_metrics(reset=True)
        final_metrics["loss"] = total_loss
    outputs = [(s, p) for s, p in zip(sentences, predictions)]
    return final_metrics, outputs