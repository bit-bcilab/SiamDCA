

import numpy as np

import keras.backend as K
from keras.callbacks import Callback


class SaveUnfrozenModel(Callback):
    def __init__(self, num_loss, log_dir, weights_name, save_layer=-1):
        super(SaveUnfrozenModel).__init__()
        self.model_index = -(num_loss * 2 + 1)
        self.log_dir = log_dir
        self.weights_name = weights_name
        self.save_layer = save_layer

    def on_epoch_end(self, epoch, logs=None):
        if self.save_layer == -1:
            model_trainable = []
            for i in range(len(self.model.layers[self.model_index].layers)):
                layer_trainable = []
                if hasattr(self.model.layers[self.model_index].layers[i], 'layers'):
                    for j in range(len(self.model.layers[self.model_index].layers[i].layers)):
                        layer_trainable.append(self.model.layers[self.model_index].layers[i].layers[j].trainable)
                        self.model.layers[self.model_index].layers[i].layers[j].trainable = True
                model_trainable.append(layer_trainable)

            self.model.layers[self.model_index].save_weights(self.log_dir + self.weights_name + str(epoch+1) + '.h5')

            for i in range(len(self.model.layers[self.model_index].layers)):
                if hasattr(self.model.layers[self.model_index].layers[i], 'layers'):
                    for j in range(len(self.model.layers[self.model_index].layers[i].layers)):
                        self.model.layers[self.model_index].layers[i].layers[j].trainable = model_trainable[i][j]
        else:
            self.model.layers[self.model_index].layers[self.save_layer].save_weights(self.log_dir + self.weights_name + str(epoch + 1) + '.h5')


class SaveUnfrozenModel_SingleGPU(Callback):
    def __init__(self, log_dir, weights_name):
        super(SaveUnfrozenModel_SingleGPU).__init__()
        self.log_dir = log_dir
        self.weights_name = weights_name

    def on_epoch_end(self, epoch, logs=None):
        model_trainable = []
        for i in range(len(self.model.layers)):
            layer_trainable = []
            if hasattr(self.model.layers[i], 'layers'):
                for j in range(len(self.model.layers[i].layers)):
                    layer_trainable.append(self.model.layers[i].layers[j].trainable)
                    self.model.layers[i].layers[j].trainable = True
            model_trainable.append(layer_trainable)

        self.model.save_weights(self.log_dir + self.weights_name + str(epoch+1) + '.h5')

        for i in range(len(self.model.layers)):
            if hasattr(self.model.layers[i], 'layers'):
                for j in range(len(self.model.layers[i].layers)):
                    self.model.layers[i].layers[j].trainable = model_trainable[i][j]


class LRStepDeCay(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, init_lr, step_per_epoch, stop_step, mid_lr, mid_step, decay_num=200, stop_lr=0.00001, verbose=0):
        super(LRStepDeCay, self).__init__()
        self.verbose = verbose
        self.r = 1.5
        self.step_per_epoch = step_per_epoch  # 2968
        self.epoch = 0
        self.decay_counter = 0
        self.losses = []
        self.real_lr = []

        self.lr = np.zeros(decay_num, dtype=np.float)
        self.lr[0] = init_lr

        a = (1 - np.power(stop_lr / init_lr, self.r * 1 / decay_num)) * (4. / np.power(decay_num, 2))
        for i in range(1, decay_num):
            self.lr[i] = self.lr[i-1] * (1 + a * i * (i - decay_num))
        self.lr[-1] = stop_lr

        start_step = 10
        mid_index = np.argmin(np.abs(self.lr - mid_lr))
        decay_step_fore = np.linspace(start_step, mid_step, mid_index+1).astype(np.int)
        decay_step_post = np.logspace(np.log(mid_step+10)/np.log(10),
                                      np.log(stop_step)/np.log(10),
                                      decay_num-mid_index-1).astype(np.int)
        self.decay_step = np.concatenate([decay_step_fore, decay_step_post]).tolist()

    def on_batch_begin(self, batch, logs=None):
        steps = self.epoch * self.step_per_epoch + batch
        if steps in self.decay_step:
            lr = self.lr[self.decay_counter]
            K.set_value(self.model.optimizer.lr, lr)
            self.decay_counter += 1
            if self.verbose > 0:
                print('\nEpoch %05d, Step %05d: LearningRateScheduler setting learning '
                      'rate to %s.' % (self.epoch + 1, batch, lr))
            logs = logs or {}
            logs['lr'] = lr

    def on_batch_end(self, batch, logs=None):
        steps = self.epoch * self.step_per_epoch + batch
        if steps in self.decay_step:
            self.losses.append(logs.get('loss'))
            self.real_lr.append(K.get_value(self.model.optimizer.lr))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch


class LRStepDeCayWarmUp(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, init_lr, step_per_epoch, stop_step, mid_lr, mid_step, warm_up=100, decay_num=200, stop_lr=1e-5):
        super(LRStepDeCayWarmUp, self).__init__()
        self.r = 1.5
        self.step_per_epoch = step_per_epoch  # 2968
        self.warm_up = warm_up
        self.epoch = 0
        self.decay_counter = 0
        self.losses = []
        self.real_lr = []

        self.warm_up_lr = np.zeros(warm_up, dtype=np.float)
        self.warm_up_lr = np.linspace(1e-6, init_lr, warm_up).astype(np.float)
        self.lr = np.zeros(decay_num, dtype=np.float)
        self.lr[0] = init_lr

        a = (1 - np.power(stop_lr / init_lr, self.r * 1 / decay_num)) * (4. / np.power(decay_num, 2))
        for i in range(1, decay_num):
            self.lr[i] = self.lr[i-1] * (1 + a * i * (i - decay_num))
        self.lr[-1] = stop_lr

        mid_index = np.argmin(np.abs(self.lr - mid_lr))
        decay_step_fore = np.linspace(warm_up+5, mid_step, mid_index+1).astype(np.int)
        decay_step_post = np.logspace(np.log(mid_step+10)/np.log(10),
                                      np.log(stop_step)/np.log(10),
                                      decay_num-mid_index-1).astype(np.int)
        self.decay_step = np.concatenate([decay_step_fore, decay_step_post]).tolist()

    def on_batch_begin(self, batch, logs=None):
        steps = self.epoch * self.step_per_epoch + batch
        if steps in self.decay_step:
            lr = self.lr[self.decay_counter]
            K.set_value(self.model.optimizer.lr, lr)
            self.decay_counter += 1
            logs = logs or {}
            logs['lr'] = lr
        elif steps < self.warm_up:
            lr = self.warm_up_lr[steps]
            K.set_value(self.model.optimizer.lr, lr)
            logs = logs or {}
            logs['lr'] = lr

    def on_batch_end(self, batch, logs=None):
        """

        Args:
            batch: 批次数
            logs:

        Returns:
            None
        """
        steps = self.epoch * self.step_per_epoch + batch
        if steps in self.decay_step:
            self.losses.append(logs.get('loss'))
            self.real_lr.append(K.get_value(self.model.optimizer.lr))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch


class LRWarmUpStepDeCayRestart(Callback):
    def __init__(self,
                 warm_up_steps,
                 restart_steps,
                 restart_num,
                 decay_steps,
                 decay_num,
                 init_lr,
                 stop_lr,
                 mid_lr,
                 mid_step,
                 loss_num,
                 unfreeze_step,
                 unfrozen_layers,
                 multi_gpu=True,
                 opt_kwargs=None):

        """
        Args:
            warm_up_steps: WarmUp所用步数
            restart_steps: 周期步数 = 学习率下降的步数 + 保持最低学习率的步数
            restart_num: 重启次数
            decay_steps: 学习发生下降的步数
            decay_num: 学习率下降的次数，设置到每10~1000步下降一次较合适，取决于总步数
            init_lr: 初始学习率
            stop_lr: 最终学习率
            mid_lr: 中值学习率
            mid_step: 中值步数
            unfreeze_step: 解冻层时的轮数
            unfrozen_layers: 解冻的层数
        """
        super(LRWarmUpStepDeCayRestart, self).__init__()
        self.r = 1.5
        self.global_step = 0
        self.epoch = 0
        self.decay_counter = 0
        self.restart_counter = 0
        self.log_lr = init_lr

        self.warm_up_steps = warm_up_steps
        self.restart_steps = restart_steps
        self.restart_num = restart_num
        self.decay_steps = decay_steps
        self.decay_num = decay_num
        self.init_lr = init_lr
        self.stop_lr = stop_lr

        self.model_index = -(loss_num * 2 + 1)
        self.backbone_index = None
        self.unfreeze_step = unfreeze_step
        self.unfrozen_layers = unfrozen_layers
        self.multi_gpu = multi_gpu
        self.opt_kwargs = opt_kwargs

        self.warm_up_lr = np.zeros(warm_up_steps, dtype=np.float)
        self.warm_up_lr = np.linspace(init_lr / 10., init_lr, warm_up_steps).astype(np.float)

        self.lr = self.count_lr(decay_num, init_lr, stop_lr)

        mid_index = np.argmin(np.abs(self.lr - mid_lr))
        decay_step_fore = np.linspace(warm_up_steps+5, warm_up_steps+mid_step-20, mid_index+1).astype(np.int)
        decay_step_post = np.logspace(np.log(warm_up_steps+mid_step+20)/np.log(10),
                                      np.log(warm_up_steps+decay_steps-20)/np.log(10),
                                      decay_num-mid_index-1).astype(np.int)
        self.decay_step = np.concatenate([decay_step_fore, decay_step_post]).tolist()

        self.restart_step = self.warm_up_steps + self.restart_steps
        self.restart_lr = self.count_lr(restart_num, init_lr, init_lr / mid_lr * stop_lr)

    def count_lr(self, decay_num, init_lr, stop_lr):
        lr = np.zeros(decay_num, dtype=np.float)
        lr[0] = init_lr
        a = (1 - np.power(stop_lr / init_lr, self.r * 1 / decay_num)) * (4. / np.power(decay_num, 2))
        for i in range(1, decay_num):
            lr[i] = lr[i - 1] * (1 + a * i * (i - decay_num))
        lr[-1] = stop_lr
        return lr

    def on_batch_begin(self, batch, logs=None):
        # WarmUp, 当步数小于设置的WarmUp步数时
        if self.global_step < self.warm_up_steps:
            lr = self.warm_up_lr[self.global_step]
            K.set_value(self.model.optimizer.lr, lr)
            self.log_lr = lr

        # Decay，当步数处于设置的衰减步数时
        elif self.global_step in self.decay_step:
            lr = self.lr[self.decay_counter]
            K.set_value(self.model.optimizer.lr, lr)
            self.decay_counter += 1
            self.log_lr = lr

        # Restart，当步数 = WarmUp结束步数 + Restart的步数 * (已经restart的次数 + 1)
        elif (self.global_step == self.restart_step) and (self.restart_counter < self.restart_num):
            # 重新计算学习率
            self.lr = self.count_lr(self.decay_num, self.restart_lr[self.restart_counter], self.stop_lr)
            # Restart计数器更新
            self.restart_counter += 1
            # 衰减次数计数器归零
            self.decay_counter = 0
            # 衰减步数设置更新
            self.decay_step = [step + self.restart_steps for step in self.decay_step]
            # Restart周期结束标志更新
            self.restart_step += self.restart_steps

        if self.global_step == self.unfreeze_step:
            if self.multi_gpu:
                if self.backbone_index is None:
                    layer_name = []
                    for layer in self.model.layers[self.model_index].layers:
                        layer_name.append(layer.name)
                    self.backbone_index = layer_name.index('Backbone')
                    self.num_layers = len(self.model.layers[self.model_index].layers[self.backbone_index].layers)

                for i in range(self.unfrozen_layers, self.num_layers):
                    self.model.layers[self.model_index].layers[self.backbone_index].layers[i].trainable = True
            else:
                if self.backbone_index is None:
                    layer_name = []
                    for layer in self.model.layers:
                        layer_name.append(layer.name)
                    self.backbone_index = layer_name.index('Backbone')
                    self.num_layers = len(self.model.layers[self.backbone_index].layers)

                for i in range(self.unfrozen_layers, self.num_layers):
                    self.model.layers[self.backbone_index].layers[i].trainable = True

            self.opt_kwargs['optimizer_kwargs'].update(dict(lr=K.get_value(self.model.optimizer.lr)))
            self.opt_kwargs['compile_kwargs'].update(
                dict(optimizer=self.opt_kwargs['optimizer'](**self.opt_kwargs['optimizer_kwargs'])))
            self.model.compile(**self.opt_kwargs['compile_kwargs'])

        logs = logs or {}
        logs['lr'] = self.log_lr
        self.global_step += 1
