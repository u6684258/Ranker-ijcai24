from .metrics import eval_f1_score, eval_admissibility, eval_interval, eval_accuracy
from .train_eval import train, evaluate

class Namespace(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.__dict__.update(kwargs)
