import json
from time import gmtime, strftime

from src.controllers.MenuController import MenuController, MODELS, ACTIONS


class MultipleRunController:

    def __init__(self, json_path, json_out_path=None, defualt_config_path='config/default_config.json'):
        self.json_out_path = json_out_path
        self.json = load_json_config(json_path)
        self.default_config = load_json_config(defualt_config_path)['global']

        self.global_config = self.json['global']
        self.instances_config = self.json['instances']

        self.stats = {
            'stats': list()
        }

    def execute(self):
        print('\nGlobal configs:')
        print('\t' + str(self.global_config), end='\n\n')
        count = 1
        for conf in self.instances_config:
            print('Instance ' + str(count) + ' configs:')
            print('\t' + str(conf))
            if 'models' in conf:
                self.execute_multiple_models(conf)
            else:
                self.execute_one_menu(conf)

            print('----------------', end='\n\n')
            count += 1

    def execute_multiple_models(self, conf):
        for model in self.get_property(self.global_config, conf, 'models'):
            print('Model:\t' + str(MODELS[model]))

            menu_args = self.prepare_menu_args(conf, model)

            menu = MenuController(**menu_args)

            self.stats['stats'].append({
                "model": MODELS[model],
                "history": menu.history,
                "score": menu.scores,
                "config": {
                    "batch_size": menu_args['batch_size'],
                    "epochs": menu_args['epochs'],
                    "image_shape": menu_args['image_shape'],
                    "color_mode": menu_args['color_mode']
                }
            })

    def execute_one_menu(self, conf):

        model = self.get_property(self.global_config, conf, 'model')
        menu_args = self.prepare_menu_args(conf, model)

        menu = MenuController(**menu_args)

        if 4 in menu_args['actions'] or 6 in menu_args['actions']:
            self.stats['stats'].append({
                "model": MODELS[model],
                "history": menu.history,
                "score": menu.scores,
                "config": {
                    "batch_size": menu_args['batch_size'],
                    "epochs": menu_args['epochs'],
                    "image_shape": menu_args['image_shape'],
                    "color_mode": menu_args['color_mode']
                }
            })

    def prepare_menu_args(self, conf, model):
        return {
            "actions": self.get_property(self.global_config, conf, 'actions'),
            "batch_size": self.get_property(self.global_config, conf, 'batch_size'),
            "color_mode": self.get_property(self.global_config, conf, 'color_mode'),
            "epochs": self.get_property(self.global_config, conf, 'epochs'),
            "image_shape": self.get_property(self.global_config, conf, 'image_shape'),
            "labels_count": self.get_property(self.global_config, conf, 'labels_count'),
            "log_folder": self.get_property(self.global_config, conf, 'log_folder'),
            "mode": 1,
            "model": model,
            "model_path": self.get_property(self.global_config, conf, 'model_path'),
            "n_train_samples": self.get_property(self.global_config, conf, 'n_train_samples'),
            "n_validation_samples": self.get_property(self.global_config, conf, 'n_validation_samples'),
            "num_workers": self.get_property(self.global_config, conf, 'num_workers'),
            "split_factor": self.get_property(self.global_config, conf, 'split_factor'),
            "test_dir": self.get_property(self.global_config, conf, 'test_dir'),
            "train_dir": self.get_property(self.global_config, conf, 'train_dir'),
            "validation_dir": self.get_property(self.global_config, conf, 'validation_dir'),
            "weights_path": self.get_property(self.global_config, conf, 'weights_path')
        }

    def get_property(self, global_config, config, key):
        if key in config:
            value = config[key]
        elif key in global_config:
            value = global_config[key]
        else:
            value = self.default_config[key]
        return value

    def save_stats_to_json(self):
        if self.json_out_path is None:
            now = strftime("%d-%m-%Y_%H-%M", gmtime())
            path = 'stats/stats_' + str(now) + '.json'
        else:
            path = self.json_out_path
        with open(path, 'w') as f:
            json.dump(self.stats, f)


def load_json_config(json_path):
    with open(json_path) as file:
        json_object = json.load(file)
    return json_object
