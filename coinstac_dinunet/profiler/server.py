from flask import Flask, send_from_directory
import os

app = Flask(__name__)


@app.route('/')
def hello():
    print(os.getcwd())
    return send_from_directory("/home/ak/TrendsLab/coinstac-dinunet/coinstac_dinunet/profiler/_profiler_stats", "test_long_file.html")

app.run()