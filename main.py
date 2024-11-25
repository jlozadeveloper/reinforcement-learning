from experiment import ExperimentRunner

def cartpole_dqn():
    return ExperimentRunner('config/dqn_cartpole.yaml')

def cartpole_dueling_dqn():
    return ExperimentRunner('config/dueling_dqn_cartpole.yaml')

def cartpole_double_dueling_dqn():
    return ExperimentRunner('config/double_dueling_dqn_cartpole.yaml')

def lunarlander_double_dueling_dqn():
    return ExperimentRunner('config/double_dueling_dqn_lunarlander.yaml')

def lunarlander_dqn():
    return ExperimentRunner('config/dqn_lunarlander.yaml')


if __name__ == "__main__":
    runner = cartpole_dqn()
    # runner = cartpole_dueling_dqn()
    # runner = cartpole_double_dueling_dqn()
    # runner = lunarlander_double_dueling_dqn()
    # runner = lunarlander_dqn()
    hyperparams = {
        ('compile/optimizer/learning_rate', 'lr'): [0.001, 0.00025],
        ('agent/gamma','dicount-gamma'): [0.99, 0.9], 
    }
    runner.train(hyperparams)