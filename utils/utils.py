import pathlib, yaml

class Config:
    @classmethod
    def project_root(cls):
        return pathlib.Path(__file__).absolute().parent.parent

    @classmethod
    def load_config(cls, config_path):
        file_path = pathlib.Path(config_path).absolute()
        if not file_path.exists():
            raise Exception(f"Config file not found: {file_path.as_posix()}")
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    @classmethod
    def get_config(cls, config:dict, return_names=False) -> dict|list:

        print(f"Config in BaseModel:")
        [print(x, y) for x, y in config.items()]
        print(f"="*20)
        if return_names:
            values = [{x:y} for x, y in config.items()]
        else:
            values = [x for x in config.values()]
        return values
        
    @classmethod
    def get_layers_config(cls, config, default_layers_config):
        dense_layers = config.get('dense_layers')
        if dense_layers is None:
            dense_layers = default_layers_config
        for k, layerConfig in enumerate(dense_layers):
            if not type(layerConfig) is dict:
                dense_layers[k] = {"units" : layerConfig, "activation": "relu"}
            else:
                if not "units" in layerConfig.keys():
                    dense_layers[k]["units"] = 64

                if not "activation" in layerConfig.keys():
                    dense_layers[k]["activation"] = "relu"
        return dense_layers