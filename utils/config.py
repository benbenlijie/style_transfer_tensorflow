import yaml


class Flag:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    @staticmethod
    def read_config(file_path):
        with open(file_path) as f:
            FLAGS = Flag(**yaml.load(f))
        return FLAGS
