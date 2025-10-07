import keras
from keras import ops


class EMA(keras.callbacks.Callback):
    def __init__(self, update_every, beta=0.9, use_for_validation=False):  # todo: use_for_validation seems to be not working
        super().__init__()
        self.beta = float(beta)
        self.update_every = int(update_every)
        self.use_for_validation = bool(use_for_validation)
        self._shadow = None
        self._backup = None
        self._step = 0
        self._n_vars = 0
        print(f"EMA model update every {update_every} steps.")

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
        if self._step % self.update_every != 0:
            return
        tv = self._ensure_slots()
        b = self.beta
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

    def on_test_begin(self, logs=None):
        if not self.use_for_validation:
            return
        self.swap_to_shadow()

    def on_test_end(self, logs=None):
        if not self.use_for_validation:
            return
        self.swap_from_shadow()


def save_ema_models(model, ema_cb, path_noema, path_ema):
    # save non EMA
    model.save(path_noema)
    # save EMA
    ema_cb.swap_to_shadow()
    try:
        model.save(path_ema)
    finally:
        ema_cb.swap_from_shadow()
