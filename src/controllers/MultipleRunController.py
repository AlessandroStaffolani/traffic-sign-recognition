import json

from src.controllers.MenuController import MenuController, MODELS, ACTIONS


class MultipleRunController:

    def __init__(self, json_path):
        with open(json_path) as file:
            self.json = json.load(file)

        self.instances_config = self.json['app']

    def execute(self):
        for conf in self.instances_config:
            print(conf)

