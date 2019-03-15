import json
from time import gmtime, strftime

from src.controllers.MenuController import MenuController, MODELS, ACTIONS


class MultipleRunController:

    def __init__(self, json_path, json_out_path=None):
        self.json_out_path = json_out_path
        with open(json_path) as file:
            self.json = json.load(file)

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
            for model in get_property(self.global_config, conf, 'models'):
                print('Model:\t' + str(MODELS[model]))

                actions = get_property(self.global_config, conf, 'actions')
                batch_size = get_property(self.global_config, conf, 'size_batch')
                epochs = get_property(self.global_config, conf, 'epochs')
                image_shape = get_property(self.global_config, conf, 'image_shape')
                num_workers = get_property(self.global_config, conf, 'n_workers')
                model_path = get_property(self.global_config, conf, 'model_path')
                weights_path = get_property(self.global_config, conf, 'weights_path')
                color_mode = get_property(self.global_config, conf, 'color_mode')
                split_factor = get_property(self.global_config, conf, 'split_factor')
                n_train_samples = get_property(self.global_config, conf, 'n_train_samples')
                n_validation_samples = get_property(self.global_config, conf, 'n_validation_samples')
                train_dir = get_property(self.global_config, conf, 'train_dir')
                validation_dir = get_property(self.global_config, conf, 'validation_dir')
                test_dir = get_property(self.global_config, conf, 'test_dir')

                menu = MenuController(mode=1,
                                      actions=actions,
                                      model=model,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      image_shape=image_shape,
                                      num_workers=num_workers,
                                      model_path=model_path,
                                      weights_path=weights_path,
                                      color_mode=color_mode,
                                      split_factor=split_factor,
                                      n_train_samples=n_train_samples,
                                      n_validation_samples=n_validation_samples,
                                      train_dir=train_dir,
                                      validation_dir=validation_dir,
                                      test_dir=test_dir
                                      )
                self.stats['stats'].append({
                    "model": MODELS[model],
                    "history": menu.history,
                    "score": menu.scores,
                    "config": {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "image_shape": image_shape,
                        "color_mode": color_mode
                    }
                })

            print('----------------', end='\n\n')
            count += 1

    def save_stats_to_json(self):
        if self.json_out_path is None:
            now = strftime("%d-%m-%Y_%H-%M", gmtime())
            path = 'stats/stats_' + str(now) + '.json'
        else:
            path = self.json_out_path
        with open(path, 'w') as f:
            json.dump(self.stats, f)


def get_property(global_object, object, key):
    try:
        value = object[key]
    except KeyError:
        value = global_object[key]
    return value

