#!/usr/bin/python
import json
import sys

from coinstac_dinunet import COINNRemote

# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)

if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    remote = COINNRemote(**args)
    remote.compute()
    remote.send()
