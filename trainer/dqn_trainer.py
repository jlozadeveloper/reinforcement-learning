import numpy as np
from .trainer_factory import register_trainer
from .base_trainer import BaseTrainer

@register_trainer('dqn')
class DQNTrainer(BaseTrainer):
    def run_episode(self, episode):
        # episode_metrics = {'reward': []}
        state, _ = self.agent.env.reset()
        done = False
        step = 0

        while not done:
            action = self.agent.select_action(np.reshape(state, [1, *self.agent.input_shape]))
            next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
            done = terminated or truncated
            batch_metrics = {'batch': step}

            policy_metrics = self.agent.update_policy(episode, action, reward)
            for metric_name, metric_value in policy_metrics.items():
                # episode_metrics.setdefault(metric_name, [])
                # episode_metrics[metric_name].append(metric_value)
                batch_metrics.setdefault(metric_name, metric_value)

            # Guardar en buffer de experiencia
            self.agent.remember(state, action, reward, next_state, terminated, truncated)
            state = next_state

            # episode_metrics['reward'].append(reward)
            batch_metrics.setdefault('reward', reward)

            # self.callbacks.on_train_batch_begin(step, {key: arr[-1] for key, arr in episode_metrics.items()})
            self.callbacks.on_train_batch_begin(step, batch_metrics)

            # Ejecutar `replay` cada cierto número de pasos
            history = self.agent.replay(episode, step)
            if history is not None:
                for metric_name, metric_value in history.history.items():
                    # episode_metrics.setdefault(metric_name, [])
                    # episode_metrics[metric_name].append(metric_value[0])
                    batch_metrics.setdefault(metric_name, metric_value[0])
            # self.callbacks.on_train_batch_end(step, {key: arr[-1] for key, arr in episode_metrics.items()})
            self.callbacks.on_train_batch_end(step, batch_metrics)
            step += 1
        # return episode_metrics
