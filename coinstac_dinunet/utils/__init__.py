import os as _os


def get_output_file_path_and_prefix(args, out_dir):
    dir_name = _os.path.join(args['state']['outputDirectory'], 'profiler_log', out_dir)
    _os.makedirs(dir_name, exist_ok=True)
    return _os.path.join(_os.path.abspath(dir_name), args['state']['clientId'])


class FrozenDict(dict):
    def __init__(self, _dict):
        super().__init__(_dict)

    def prompt(self, key, value):
        raise ValueError(f'*** '
                         f'Attempt to modify frozen dict '
                         f'[{key} : {self[key]}] with [{key} : {value}]'
                         f' ***')

    def __setitem__(self, key, value):
        if key not in self:
            super(FrozenDict, self).__setitem__(key, value)
        else:
            self.prompt(key, value)

    def update(self, **kw):
        for k, v in kw.items():
            self[k] = v
