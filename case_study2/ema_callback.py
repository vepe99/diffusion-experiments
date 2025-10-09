import keras
from keras import ops


class EMA(keras.callbacks.Callback):
    """ Exponential Moving Average (EMA) callback for Keras models.

    Following the ideas from
    * Karras et. al., https://arxiv.org/pdf/2312.02696
    """
    def __init__(self, gamma=6.94):
        super().__init__()
        self._shadow = None
        self._backup = None
        self._step = 0
        self._n_vars = 0
        self._gamma = gamma
        # sigma_rel = (gamma + 1)**(0.5) * (gamma +2)**(-1) * (gamma + 3)**(-0.5) = 10%
        print(f"Using EMA for training.")

    def beta(self, t):
        return (1 - 1/t)**(self._gamma + 1)

    def _snapshot(self, v):
        t = ops.convert_to_tensor(v)
        try:
            t = ops.stop_gradient(t)
        except Exception as e:
            print(e)
        return t

    def _init_slots_from_tv(self, tv):
        self._shadow = [self._snapshot(v) for v in tv]
        self._backup = [None] * len(tv)
        self._n_vars = len(tv)

    def _ensure_slots(self):
        tv = self.model.trainable_variables
        if self._shadow is None or self._n_vars != len(tv):
            self._init_slots_from_tv(tv)
        return tv

    def on_train_begin(self, logs=None):
        self._step = 0
        self._ensure_slots()

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1
        tv = self._ensure_slots()
        b = self.beta(self._step)
        new_shadow = []
        for s, v in zip(self._shadow, tv):
            v_now = self._snapshot(v)
            if ops.dtype(s) != ops.dtype(v_now):
                v_now = ops.cast(v_now, ops.dtype(s))
            new_shadow.append(b * s + (1.0 - b) * v_now)
        self._shadow = new_shadow

    def swap_to_shadow(self):
        tv = self._ensure_slots()
        for i, v in enumerate(tv):
            self._backup[i] = self._snapshot(v)
            w = self._shadow[i]
            if ops.dtype(w) != v.dtype:
                w = ops.cast(w, v.dtype)
            v.assign(w)

    def swap_from_shadow(self):
        tv = self._ensure_slots()
        for i, v in enumerate(tv):
            v.assign(self._backup[i])
        self._backup = [None] * len(tv)


def save_ema_models(model, ema_cb, path_noema, path_ema):
    # save non EMA
    model.save(path_noema)
    # save EMA
    ema_cb.swap_to_shadow()
    try:
        model.save(path_ema)
    finally:
        ema_cb.swap_from_shadow()
e