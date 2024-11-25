import numpy as np, tensorflow as tf, time, pathlib
from keras.src import callbacks as callbacks_module
from agent import BaseAgent
from tensorboard.plugins.hparams import api as hp
from utils import callback as custom_callback

class BaseTrainer:
    def __init__(self, agent:BaseAgent, **config):
        self.agent = agent
        self.num_episodes = config.get('num-episodes', 100)
        self.logs_folder:pathlib.Path = pathlib.Path(config.get('logs-folder'))
        self.hyperparams = config.get('hyperparams', None)
        callbacks = config.get('callbacks', [])
        callbacks = [
            custom_callback.GenericMetrics(['epsilon','rwrd'], ['epsilon','reward'], [np.amin, np.sum]),
            # custom_callback.EpisodeReward(),
            custom_callback.NEpisodesReward('50_ep_rwrd', 50, np.mean, np.sum),
            custom_callback.MetricsUpdater(),
            custom_callback.MetricsPrinter(['Ep', 'Epsilon', 'Rwrd', '50_Ep_Rwrd'],['episode', 'epsilon', 'rwrd', '50_ep_rwrd']),
            callbacks_module.ModelCheckpoint(
                (self.logs_folder / 'checkpoints/checkpoint_{epoch}.keras').as_posix(),
                monitor='rwrd',
                verbose=0,
                save_best_only=True,
                mode="max",
            ),
            callbacks_module.TensorBoard(
                self.logs_folder.as_posix(),
                histogram_freq=5,
                update_freq='epoch',
                write_steps_per_second=True
            )
        ]
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=False,
                epochs=self.num_episodes,
                model=self.agent.model,
            )
        self.callbacks:callbacks_module.CallbackList = callbacks
        self.config = config

    def train(self):
        # tf.data.experimental.enable_debug_mode()
        # tf.config.run_functions_eagerly(True)
        print(f'Executing eagerly: {tf.executing_eagerly()}')
        print(f'Executing functions eagerly: {tf.config.functions_run_eagerly()}')
        self.callbacks.on_train_begin()
        with tf.summary.create_file_writer((self.logs_folder / 'train').as_posix()).as_default():
            hp.hparams(self.hyperparams)
        train_metrics = {}
        for episode in range(self.num_episodes):
            ep_metrics = {'episode': episode}
            self.callbacks.on_epoch_begin(episode)
            self.run_episode(episode)
            # self.log_metrics(episode, episode_metrics, train_metrics)
            # self.callbacks.on_epoch_end(episode, {key: arr[-1] for key, arr in train_metrics.items()})
            self.callbacks.on_epoch_end(episode, ep_metrics)

            self.agent.episode_ended(episode, ep_metrics)
            
        # Plotting u otras acciones    
        self.plot_metrics(train_metrics)

    def run_episode(self, episode):
        """Define un episodio de entrenamiento. Método para ser implementado en subclases."""
        raise NotImplementedError

    def log_metrics(self, episode, episode_metrics, train_metrics):
        """Actualiza métricas por episodio y las muestra en consola."""
        for metric_name, metric_values in episode_metrics.items():
            if metric_name not in train_metrics:
                train_metrics[metric_name] = []
            if metric_name == 'reward':
                train_metrics[metric_name].append(np.sum(metric_values))
            elif metric_name == 'epsilon':
                train_metrics[metric_name].append(np.amin(metric_values))
            else:
                train_metrics[metric_name].append(np.mean(metric_values))
        train_metrics.setdefault('100_ep_rewards',[])
        train_metrics['100_ep_rewards'].append(np.mean(train_metrics['reward'][-100:]))
        print(f"Episode: {episode}", end='')
        [print(f", {metric_name}: {value[-1]:.4f}", end='') for metric_name, value in train_metrics.items()]
        print('')