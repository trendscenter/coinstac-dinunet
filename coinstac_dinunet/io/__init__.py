import sys
import json

RECV = sys.stdin.read()
try:
    RECV = json.loads(RECV)
except:
    print('Invalid json input:', RECV)
