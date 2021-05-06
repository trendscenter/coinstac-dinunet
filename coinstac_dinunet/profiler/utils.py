import io
import os

from pyinstrument import renderers


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