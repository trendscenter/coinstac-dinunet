import functools
from pyinstrument import Profiler, renderers
import os
import json
import datetime
import glob
import sys
import argparse
from coinstac_dinunet.profiler.utils import JSONToHTML
from coinstac_dinunet.config import profiler_conf_file


def boolean_string(s):
    try:
        return str(s).strip().lower() == 'true'
    except:
        return False


default_args = {}
if '--profile' in sys.argv and boolean_string(sys.argv[sys.argv.index('--profile') + 1]):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=False, type=boolean_string, help="Run Profiler.")
    ap.add_argument("--profiler_gather_freq", default=1, type=int,
                    help="Frequency to gather profiler data.")
    ap.add_argument("--profiler_verbose", default=False, type=boolean_string, help="Verbose.")
    ap.add_argument("--profiler_dir_key", default='outputDirectory', type=str, help="Profiler log directory.")
    _args, _ = ap.parse_known_args()
    default_args = vars(_args)


class Conf:
    enabled = False
    log_dir = None
    verbose = False
    gather_freq = False


class Profile:
    _GATHER_KEYS_ = ['duration', 'sample_count', 'cpu_time']
    _DATE_FMT_ = '%m/%d/%y %H:%M:%S'

    def __init__(self, conf: Conf = None, **kw):
        if conf is None:
            self.enabled = default_args.get('profile', False)
            self.log_dir = default_args.get('profiler_log_dir')
            self.verbose = default_args.get('profiler_verbose', False)
            self.gather_frequency = default_args.get('profiler_gather_freq', 1)
            if self.enabled:
                conf_file = profiler_conf_file
                if os.path.exists(conf_file):
                    conf = json.loads(open(conf_file).read())
                    self.log_dir = conf['log_dir']
                else:
                    self.enabled = False

                if self.log_dir is not None:
                    os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.enabled = conf.enabled
            self.log_dir = conf.log_dir
            self.verbose = conf.verbose
            self.gather_frequency = conf.gather_freq

    def __call__(self, func):
        if self.verbose:
            print("*** Profiling ***", func, f"Enabled: {self.enabled}")
        if not self.enabled:
            return func

        @functools.wraps(func)
        def call(*args, **kwargs):
            stats_htm = f"{self.log_dir}{os.sep}{func.__name__}_STATS.html"
            stats_json = f"{self.log_dir}{os.sep}{func.__name__}_STATS.json"
            _stats_json = f"{self.log_dir}{os.sep}_{func.__name__}_PART_.json"

            profiler = Profiler()
            profiler.start()
            ret = func(*args, **kwargs)
            profiler.stop()
            jsns = [profiler.output(renderers.JSONRenderer())]

            with open(_stats_json, 'w', encoding='utf-8') as file:
                file.write(jsns[0])

            files = glob.glob(self.log_dir + f"{os.sep}*{func.__name__}_PART_*.json")
            if len(files) % self.gather_frequency == 0:

                jsns = [json.load(open(jsn, encoding='utf-8')) for jsn in files]
                trip_time = [j['start_time'] for j in jsns]
                start = datetime.datetime.fromtimestamp(min(trip_time)).strftime(Profile._DATE_FMT_)
                end = datetime.datetime.fromtimestamp(max(trip_time)).strftime(Profile._DATE_FMT_)

                start = datetime.datetime.strptime(start, Profile._DATE_FMT_)
                end = datetime.datetime.strptime(end, Profile._DATE_FMT_)
                trip_duration = str((end - start) / self.gather_frequency)

                if os.path.exists(stats_json):
                    jsns.append(json.load(open(stats_json, encoding='utf-8')))

                jsn = self._gather(jsns)
                with open(stats_json, 'w', encoding='utf-8') as file:
                    file.write(jsn)

                renderer = JSONToHTML(
                    json_str=jsn,
                    trip_sample_size=self.gather_frequency,
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
