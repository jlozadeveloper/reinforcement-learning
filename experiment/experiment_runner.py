import gymnasium as gym, itertools, copy, pathlib, time
from keras import optimizers, metrics, losses, Model
from utils import Config, metrics as CustomMetrics
from model import ModelFactory
from policy import PolicyFactory
from replay_buffer import ReplayBufferFactory, BaseReplayBuffer
from agent import  AgentFactory
from trainer import TrainerFactory


class ExperimentRunner:
    def __init__(self, config_path):
        self.config:dict = Config.load_config(config_path)

        self.env = None
        self.model = None
        self.policy = None
        self.replay_buffer = None
        self.agent = None
        self.trainer = None

        self.base_folder:pathlib.Path = pathlib.Path('data/tensorboard')

    def _setup(self, config = None):
        if config is None:
            config = self.config.copy()

        # environment
        env_config:dict = config.get('environment', {})
        self.logs_folder = self.base_folder / env_config.get('id','unkown_env')
        self.env:gym.Env = gym.make(**env_config)
        # video_folder = "./data"
        # env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: episode_id % 1 == 0,) # Guardar video cada 10 episodios
        # env = gym.wrappers.RecordEpisodeStatistics(env)

        input_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n

        # model
        # model_config = Config.get_config_vars(config, 'model')
        model_config = config.get('model',{})
        self.logs_folder = self.logs_folder / model_config.get('classname','unkown_model')
        model_config['input_shape'] = input_shape
        model_config['num_actions'] = num_actions
        self.model: Model = ModelFactory.create_model(**model_config)

        # compile
        compile_config:dict = config.get('compile', {})
            # loss
        loss_config = compile_config.get('loss',{'function': 'MeanSquaredError'})
        loss_fn = loss_config.pop("function")
        loss_class = getattr(losses, loss_fn, None)  # Get loss class from tf.keras.losses
        if not loss_class:
            raise ValueError(f"Unknown loss function class: {loss_fn}")
            # optimizer
        optimizer_config = compile_config.get('optimizer',{'classname': 'Adam'})
        optimizer_class_name = optimizer_config.pop("classname")
        optimizer_class = getattr(optimizers, optimizer_class_name, None)  # Get loss class from tf.keras.losses
        if not optimizer_class:
            raise ValueError(f"Unknown optimizer class: {optimizer_class_name}")
            # metrics
        metrics_config:tuple[dict] = compile_config.get('metrics', [])
        metrics_array = []
        metrics_namespaces = [metrics, CustomMetrics]
        for metric_config in metrics_config:
            metric_class_name = metric_config.get('class','')
            for namespace in metrics_namespaces:
                metric_class = getattr(namespace, metric_class_name, None)  # Get metric class from tf.keras.metrics
                if metric_class:
                    metric_params = metric_config.get('params',{})
                    metric_instance = metric_class(**metric_params)
                    metrics_array.append(metric_instance)
                    break
            if metric_class is None:
                raise Warning(f"Metric class '{metric_class_name}' not found.")
        
        self.model.compile(loss=loss_class(**loss_config), optimizer=optimizer_class(**optimizer_config), metrics=metrics_array)
        self.model.summary()


        # policy
        # policy_config = Config.get_config_vars(config, "policy")
        policy_config = config.get("policy", {})
        policy_config['num_actions'] = num_actions
        self.policy = PolicyFactory.create_policy(**policy_config)

        # replay buffer
        replay_config = config.get("replay-buffer", {})
        self.replay_buffer:BaseReplayBuffer = ReplayBufferFactory.create_buffer(**replay_config)

        # agent
        agent_config = config.get("agent", {})
        self.logs_folder = self.logs_folder / agent_config.get('classname','unkown_agent')
        agent_config['env'] = self.env
        agent_config['model'] = self.model
        agent_config['policy'] = self.policy
        agent_config['replay_buffer'] = self.replay_buffer
        # agent = DQNAgent(env, model, policy, replay, agent_config)
        self.agent = AgentFactory.create_agent(**agent_config)

        # Crear instancias de Trainer y Evaluator
        trainer_config = config.get("train", {})
        trainer_config['agent'] = self.agent
        trainer_config['logs-folder'] = (self.logs_folder / time.strftime('%m%d_%H%M')).absolute().as_posix()
        self.trainer = TrainerFactory.create_trainer(**trainer_config)
        
        # evaluator = Evaluator(agent, num_episodes=5)        
        # # Crear trainer según el tipo especificado en la configuración

    def _get_hyperparams_combinations(self, hyperparams):
        return list(itertools.product(*{k: v for k, v in hyperparams.items()}.values()))

    def _apply_config_params(self, hyperparams, params):
        def nested_set(dic, path, value):
            keys = path[0].split('/')
            for key in keys[:-1]:
                dic = dic.setdefault(key, {})
            dic[keys[-1]] = value

        merged_config = copy.deepcopy(self.config)
        for key, path in enumerate(hyperparams):
            nested_set(merged_config, path, params[key])
        return merged_config
    
    def train(self, hyperparams = None):
        if hyperparams is None:
            self._setup()
            self.trainer.train()
            return
        
        hyperparams_combinations = self._get_hyperparams_combinations(hyperparams)
        for combo in hyperparams_combinations:
            execution_config = self._apply_config_params(hyperparams, combo)
            execution_hyperparams = {val[1]: combo[key] for key, val in enumerate(hyperparams)}
            execution_config['train']['hyperparams'] = execution_hyperparams
            self._setup(execution_config)
            self.trainer.train()
        

    def evaluate(self):
        # self.evaluator.evaluate()
        pass