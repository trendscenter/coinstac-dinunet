from coinstac_dinunet.config import ProfilerConf

ProfilerConf.enabled = True

import functools
from pyinstrument import Profiler, renderers
import os
import json
import io
import datetime


class JSONToHTML(renderers.HTMLRenderer):
    def __init__(self, json_str=None, trip_duration=None, trip_sample_size=None, **kw):
        super().__init__(**kw)
        self.json_str = json_str
        self.trip_duration = trip_duration
        self.trip_sample_size = trip_sample_size

    def set_json(self, json_str):
        self.json_str = json_str

    def render_json(self, session):
        if self.json_str is not None:
            return self.json_str
        return super().render_json(session)

    def render(self, session):
        resources_dir = os.path.join(os.path.dirname(os.path.abspath(renderers.__file__)), 'html_resources/')

        if not os.path.exists(os.path.join(resources_dir, 'app.js')):
            raise RuntimeError("Could not find app.js. If you are running "
                               "pyinstrument from a git checkout, run 'python "
                               "setup.py build' to compile the Javascript "
                               "(requires nodejs).")

        with io.open(os.path.join(resources_dir, 'app.js'), encoding='utf-8') as f:
            js = f.read()

        session_json = self.render_json(session)

        trip_duration_bar_style = "text-align:right;background-color:#212729;font-family: 'Lucida Console', 'Courier New', monospace;font-size:18px;padding:9px 30px 9px 9px"
        page = u'''<!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
            </head>
            <body>
                <div id="trip-stats" style="{trip_duration_style}">COINSTAC iteration(#samples={trip_samples}): <span style="color:#d4414d">{trip_duration}</span></div>
                <div id="app"></div>
                <script>
                    window.profileSession = {session_json}
                </script>
                <script>
                    {js}
                </script>
            </body>
            </html>'''.format(
            trip_duration_style=trip_duration_bar_style,
            trip_samples=self.trip_sample_size,
            trip_duration=self.trip_duration,
            js=js, session_json=session_json
        )

        return page


import glob


class Profile:
    _GATHER_KEYS_ = ['duration', 'sample_count', 'cpu_time']
    _DATE_FMT_ = '%m/%d/%y %H:%M:%S'

    def __init__(self, conf=ProfilerConf, **kw):
        self.conf = conf
        self.verbose = kw.get('verbose', False)
        os.makedirs(conf.log_dir, exist_ok=True)

    def __call__(self, func):
        if self.verbose:
            print("*** Profiling ***", func, self.conf.enabled)
        if not self.conf.enabled:
            return func

        @functools.wraps(func)
        def call(*args, **kwargs):
            stats_htm = f"{self.conf.log_dir}{os.sep}{func.__name__}_STATS.html"
            stats_json = f"{self.conf.log_dir}{os.sep}{func.__name__}_STATS.json"
            _stats_json = f"{self.conf.log_dir}{os.sep}_{func.__name__}_PART_.json"

            profiler = Profiler()
            profiler.start()
            ret = func(*args, **kwargs)
            profiler.stop()
            jsns = [profiler.output(renderers.JSONRenderer())]

            with open(_stats_json, 'w', encoding='utf-8') as file:
                file.write(jsns[0])

            files = glob.glob(self.conf.log_dir + f"{os.sep}*{func.__name__}_PART_*.json")
            if len(files) % self.conf.gather_frequency == 0:

                jsns = [json.load(open(jsn, encoding='utf-8')) for jsn in files]
                trip_time = [j['start_time'] for j in jsns]
                start = datetime.datetime.fromtimestamp(min(trip_time)).strftime(Profile._DATE_FMT_)
                end = datetime.datetime.fromtimestamp(max(trip_time)).strftime(Profile._DATE_FMT_)

                start = datetime.datetime.strptime(start, Profile._DATE_FMT_)
                end = datetime.datetime.strptime(end, Profile._DATE_FMT_)
                trip_duration = str((end - start) / self.conf.gather_frequency)

                if os.path.exists(stats_json):
                    jsns.append(json.load(open(stats_json, encoding='utf-8')))

                jsn = self._gather(jsns)
                with open(stats_json, 'w', encoding='utf-8') as file:
                    file.write(jsn)

                renderer = JSONToHTML(
                    json_str=jsn,
                    trip_sample_size=self.conf.gather_frequency,
                    trip_duration=trip_duration
                )

                with open(stats_htm, 'w', encoding='utf-8') as file:
                    file.write(profiler.output(renderer))

                [os.remove(f) for f in files]
            return ret

        return call

    def _gather(self, jsons):
        dest = jsons[-1]
        for j in jsons[:-1]:
            for k in Profile._GATHER_KEYS_:
                dest[k] += j[k]
        return json.dumps(dest)


class Test:
    @Profile(conf=ProfilerConf)
    def test_long(self):
        # self.test1()
        # self.test2()
        for i in range(10000):
            a = [0] * 1000
        for j in range(28303):
            b = 0 * [10000]

    @Profile(conf=ProfilerConf)
    def test1(self):
        self.test3()
        for i in range(10100):
            a = [0] * 1000
        for j in range(2833):
            b = 0 * [10000]

    def test2(self):
        for i in range(10100):
            a = [0] * 1000
        for j in range(2833):
            b = 0 * [10000]

    def test3(self):
        for i in range(10100):
            a = [0] * 1000
        for j in range(2833):
            b = 0 * [10000]


r"""Testing"""
if __name__ == "__main__":
    for i in range(100):
        c = Test()
        c.test_long()
        c.test1()

        # @Profile(conf=ProfilerConf)
        # def test_long():
        #     test1()
        #     test2()
        #     for i in range(10000):
        #         a = [0] * 1000
        #     for j in range(28303):
        #         b = 0 * [10000]
        #
        #
        # def test1():
        #     test3()
        #     for i in range(10100):
        #         a = [0] * 1000
        #     for j in range(2833):
        #         b = 0 * [10000]
        #
        #
        # def test2():
        #     for i in range(10100):
        #         a = [0] * 1000
        #     for j in range(2833):
        #         b = 0 * [10000]
        #
        #
        # def test3():
        #     for i in range(101000):
        #         a = [0] * 1000
        #     for j in range(2833):
        #         b = 0 * [10000]
