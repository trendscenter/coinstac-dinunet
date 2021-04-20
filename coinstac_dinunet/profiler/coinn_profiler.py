import functools
from pyinstrument import Profiler, renderers
import os
import json
from coinstac_dinunet.config import ProfilerConf
import copy


# from .server import app


class JSONToHTML(renderers.HTMLRenderer):
    def __init__(self, json_str=None, **kw):
        super().__init__(**kw)
        self.json_str = json_str

    def set_json(self, json_str):
        self.json_str = json_str

    def render_json(self, session):
        if self.json_str is not None:
            return self.json_str
        return super().render_json(session)


import glob


class Profile:
    _GATHER_KEYS_ = ['duration', 'sample_count', 'cpu_time']

    def __init__(self, conf=ProfilerConf, renderer=JSONToHTML(), **kw):
        self.conf = conf
        os.makedirs(conf.log_dir, exist_ok=True)
        self.renderer = renderer

    def __call__(self, func):
        if not self.conf.enabled:
            return func

        @functools.wraps(func)
        def call(*args, **kwargs):
            stats_htm = f"{self.conf.log_dir}{os.sep}{func.__name__}_STATS.html"
            stats_json = f"{self.conf.log_dir}{os.sep}{func.__name__}_STATS.json"
            _stats_json = f"{self.conf.log_dir}{os.sep}_{func.__name__}_{self.conf.iter % self.conf.gather_frequency}.json"

            profiler = Profiler()
            profiler.start()
            ret = func(*args, **kwargs)
            profiler.stop()
            jsns = [profiler.output(renderers.JSONRenderer())]

            with open(_stats_json, 'w', encoding='utf-8') as file:
                file.write(jsns[0])

            if self.conf.iter % self.conf.gather_frequency == 0:
                jsns = [json.load(open(jsn, encoding='utf-8')) for jsn in glob.glob(self.conf.log_dir + "/_*.json")]

                if os.path.exists(stats_json):
                    jsns.append(json.load(open(stats_json, encoding='utf-8')))

                jsn = self._gather(jsns)
                with open(stats_json, 'w', encoding='utf-8') as file:
                    file.write(jsn)

                renderer = self.renderer
                renderer.set_json(jsn)
                with open(stats_htm, 'w', encoding='utf-8') as file:
                    file.write(profiler.output(renderer))
            return ret

        return call

    def _gather(self, jsons):

        dest = jsons[-1]
        for j in jsons[:-1]:
            for k in Profile._GATHER_KEYS_:
                dest[k] += j[k]

        return json.dumps(dest)


if __name__ == "__main__":
    ProfilerConf.gather_frequency = 1
    ProfilerConf.enabled = True


    @Profile(ProfilerConf)
    def test_long():
        test1()
        test2()
        for i in range(1000):
            a = [0] * 1000
        for j in range(2833):
            b = 0 * [10000]


    def test1():
        test3()
        for i in range(10100):
            a = [0] * 1000
        for j in range(2833):
            b = 0 * [10000]


    def test2():
        for i in range(10100):
            a = [0] * 1000
        for j in range(2833):
            b = 0 * [10000]


    def test3():
        for i in range(10100):
            a = [0] * 1000
        for j in range(2833):
            b = 0 * [10000]


    test_long()
