import json
import os
from types import SimpleNamespace

import yaml


def load_config(path, data_dir=None, output_dir=None):
    with open(path, encoding='utf-8') as file:
        values = yaml.safe_load(file)

    if data_dir:
        values['data_dir'] = data_dir
    if output_dir:
        values['output_dir'] = output_dir

    with open(os.path.join(values['data_dir'], 'entity2id.json'), encoding='utf-8') as file:
        values['num_entities'] = len(json.load(file))
    with open(os.path.join(values['data_dir'], 'relation2id.json'), encoding='utf-8') as file:
        values['num_relations'] = len(json.load(file))
    return SimpleNamespace(**values)
