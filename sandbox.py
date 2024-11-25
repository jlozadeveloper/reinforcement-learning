import numpy as np, gymnasium as gym, itertools, yaml, tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from evaluator.evaluator import Evaluator
# from trainer.trainer import Trainer
from utils import Config
from model import MPLModel
from replay_buffer import ReplayBufferFactory, BaseReplayBuffer
from policy import PolicyFactory
from agent import DQNAgent
from keras import optimizers, metrics, losses, models
from experiment import ExperimentRunner
from replay_buffer.episodic_replay_buffer import EpisodicReplayBuffer


def running_old():
    # config
    config_path = './config/dqn_cartpole.yaml'
    config:dict = Config.load_config(config_path)

    # environment
    env_config = config.get('environment')
    env_name = env_config.pop("name")
    env:gym.Env = gym.make(env_name, **env_config)
    video_folder = "./data"
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: episode_id % 1 == 0,) # Guardar video cada 10 episodios
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # model
    model_config = config.get('model')
    model = MPLModel.create_model(input_shape, num_actions, model_config)

    # metricas
    metrics_config:dict = config.get('metrics', [])
    metrics_array = []
    for metric_config in metrics_config:  # Handle missing 'metrics' key
        metric_class_name = metric_config.get('class','')
        metric_class = getattr(metrics, metric_class_name, None)  # Get metric class from tf.keras.metrics
        if metric_class:
            metric_params = metric_config.get('params',{})
            metric_instance = metric_class(**metric_params)
            metrics_array.append(metric_instance)
        else:
            print(f"Warning: Metric class '{metric_class_name}' not found.")

    # compile
    model = models.load_model('cartpole-dqn.keras')
    # model.compile(loss="mse", optimizer=optimizers.RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.compile(loss=losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=0.00025, epsilon=0.01), metrics=metrics_array)
    model.summary()


    # policy
    policy_config = config.get("policy","")
    policy = PolicyFactory.create_policy(**policy_config)

    # replay buffer
    replay_config = config.get("replay-buffer")
    replay:BaseReplayBuffer = ReplayBufferFactory.create_buffer(**replay_config)

    # agent
    agent_config = config.get("agent")
    agent = DQNAgent(env, model, policy, replay, agent_config)

    # Crear instancias de Trainer y Evaluator
    trainer = Trainer(agent, num_episodes=1000, config=config)
    evaluator = Evaluator(agent, num_episodes=5)

    # Entrenar el agente
    trainer.train()

    # evaluar el modelo
    evaluator.evaluate()

def running_new():
    # runner = ExperimentRunner('config/dqn_cartpole.yaml')
    runner = ExperimentRunner('config/dueling_dqn_cartpole.yaml')
    runner.run()

def run_test():
    # config
    config_path = './config/dqn_cartpole.yaml'
    config:dict = Config.load_config(config_path)

    # environment
    env_config = config.get('environment')
    env_name = env_config.pop("name")
    env:gym.Env = gym.make(env_name, **env_config)

    buffer = EpisodicReplayBuffer(50)
    for i in range(50):
        state, _ = env.reset()
        done = False
        step = 0

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Guardar en buffer de experiencia
            buffer.add(state, action, reward, next_state, terminated, truncated)
            batch = buffer.sample(min(len(buffer),2))
            state = next_state
            step += 1

def nested_set(dic, path, value):
    keys = path.split('/')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def test_summary():
    with tf.summary.create_file_writer('./data/tensorboard/jlo_run_4').as_default():
        hp.hparams({
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        })

def test_itertools():
    hyperparams = {
        'compile/optimizer/learning_rate': [0.001, 0.00025],
        'agent/gamma': [0.99, 0.9]
    }

    config_path = './config/dqn_cartpole.yaml'
    config:dict = Config.load_config(config_path)

    # for key in hyperparams.keys():
    #     hyperparams[key.split('/')] = hyperparams.pop(key)

    combinations = list(itertools.product(*{tuple(k.split('/')): v for k, v in hyperparams.items()}.values()))

    for combo in combinations:
        current_config = config.copy()
        for i, path in enumerate(hyperparams):
            nested_set(current_config, path, combo[i])

def test_play():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env.metadata['render_fps'] = 8
    from gymnasium.utils import play
    play.play(env, zoom=1, keys_to_action={"a": np.array(0, dtype=np.int64),
        "d": np.array(1, dtype=np.int64),})
        


# running_new()
# run_test()
# test_itertools()

test_play()