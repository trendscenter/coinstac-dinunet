import functools
import time

from pyinstrument import Profiler, renderers
import os
import json
import datetime
import glob
from coinstac_dinunet.profiler.utils import JSONToHTML
from coinstac_dinunet.config import default_args as _args
from coinstac_dinunet.io import RECV as _RECV
import uuid as _UID


class Profile:
    _GATHER_KEYS_ = ['duration', 'sample_count', 'cpu_time']
    _DATE_FMT_ = '%m/%d/%y %H:%M:%S'

    def __init__(self, **kw):
        self.enabled = _args.get('profile', False)
        self.log_dir = _args.get('profiler_log_dir')
        self.verbose = _args.get('profiler_verbose', False)
        self.gather_frequency = _args.get('profiler_gather_freq', 1)
        if self.enabled:
            self.log_dir = _RECV['state']['outputDirectory'] + os.sep + "_profiler_logs"
            if self.log_dir is not None:
                os.makedirs(self.log_dir, exist_ok=True)

    def __call__(self, func):
        if self.verbose:
            print("*** Profiling ***", func, f"Enabled: {self.enabled}")
        if not self.enabled:
            return func

        @functools.wraps(func)
        def call(*args, **kwargs):
            stats_htm = f"{self.log_dir}{os.sep}{func.__name__}_STATS.html"
            stats_json = f"{self.log_dir}{os.sep}{func.__name__}_STATS.json"
            _stats_json = f"{self.log_dir}{os.sep}_{func.__name__}_PART_{_UID.uuid4()}.json"

            profiler = Profiler()
            profiler.start()
            ret = func(*args, **kwargs)
            profiler.stop()
            jsns = [profiler.output(renderers.JSONRenderer())]

            with open(_stats_json, 'w', encoding='utf-8') as file:
                file.write(jsns[0])

            files = glob.glob(self.log_dir + f"{os.sep}*{func.__name__}_PART_*.json")
            if len(files) % (self.gather_frequency + 1) == 0:
                jsns = [json.load(open(jsn, encoding='utf-8')) for jsn in files]
                start_times = [j['start_time'] for j in jsns]
                comp_duration = sum(
                    [datetime.timedelta(0, j['duration']).total_seconds() for j in jsns][:self.gather_frequency]
                )
                comp_duration = datetime.timedelta(0, comp_duration)

                start = datetime.datetime.fromtimestamp(min(start_times))
                end = datetime.datetime.fromtimestamp(max(start_times))

                trip_duration = end - start - comp_duration
                avg_trip_duration = str(trip_duration / self.gather_frequency)

                jsn = self._gather(jsns[:self.gather_frequency])

                _current = {}
                if os.path.exists(stats_json):
                    _current = json.loads(open(stats_json).read())

                jsn['start_time'] = _current.get('start_time', time.time())
                jsn['time_elapsed'] = str(datetime.datetime.fromtimestamp(time.time()) - datetime.datetime.fromtimestamp(
                    jsn['start_time']))

                jsn['total_seconds_in_trip'] = trip_duration.total_seconds() + _current.get(
                    'total_seconds_in_trip',
                    0)
                jsn['total_seconds_in_computation'] = comp_duration.total_seconds() + _current.get(
                    'total_seconds_in_computation',
                    0)
                jsn['avg_trip_duration'] = avg_trip_duration
                with open(stats_json, 'w', encoding='utf-8') as file:
                    file.write(json.dumps(jsn))

                renderer = JSONToHTML(
                    json=jsn,
                    trip_sample_size=self.gather_frequency
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
        for k in Profile._GATHER_KEYS_:
            dest[k] = dest[k] / len(jsons)
        return dest
