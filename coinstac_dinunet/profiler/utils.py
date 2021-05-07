import io
import os
import json

from pyinstrument import renderers
import datetime


class JSONToHTML(renderers.HTMLRenderer):
    def __init__(self, json=None, trip_sample_size=None, **kw):
        super().__init__(**kw)
        self.json = json
        self.trip_sample_size = trip_sample_size

    def render_json(self, session):
        if self.json is not None:
            return json.dumps(self.json)
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

        avg_trip_css = "text-align:right;background-color:#21181b;font-family: 'Lucida Console', 'Courier New', monospace;font-size:18px;padding:9px 30px 9px 9px"
        duration_css = "text-align:right;background-color:#000000;font-family: 'Lucida Console', 'Courier New', monospace;font-size:18px;padding:9px 30px 9px 9px"
        page = u'''<!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
            </head>
            <body>
                <div id="trip-stats" style="{avg_trip_css}">Total Time Elapsed: <span style="color:#d4414d">{time_elapsed}</span></div>
                <div id="trip-stats" style="{avg_trip_css}">Average trip duration(n={trip_samples}): <span style="color:#d4414d">{trip_duration}</span></div>
                <div id="trip-stats" style="{duration_css}">Total time in trips: <span style="color:#d4414d">{total_duration_in_trip}</span></div>
                <div id="trip-stats" style="{duration_css}">Total time in computation: <span style="color:#d4414d">{total_duration_in_computation}</span></div>
                <div id="app"></div>
                <script>
                    window.profileSession = {session_json}
                </script>
                <script>
                    {js}
                </script>
            </body>
            </html>'''.format(
            time_elapsed=str(self.json['time_elapsed']),
            avg_trip_css=avg_trip_css,
            duration_css=duration_css,
            trip_samples=self.trip_sample_size,
            trip_duration=self.json['avg_trip_duration'],
            total_duration_in_trip=str(datetime.timedelta(0, self.json['total_seconds_in_trip'])),
            total_duration_in_computation=str(datetime.timedelta(0, self.json['total_seconds_in_computation'])),
            js=js, session_json=session_json
        )

        return page
