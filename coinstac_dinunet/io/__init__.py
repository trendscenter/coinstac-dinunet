#!/usr/bin/env python

"""
Forked from https://pypi.org/project/coinstac/
"""

import asyncio as _asyncio
import websockets as _ws
import json as _json
from coinstac_dinunet.utils import duration as _duration
from coinstac_dinunet.utils.logger import *
import time as _time


class COINPyService:
    def __init__(self, **kw):
        self.cache = kw.get('cache', {})
        self.debug = kw.get('debug', True)

    def _local(self, msg) -> callable:
        return ...

    def _remote(self, msg) -> callable:
        return ...

    def _local_compute_args(self, msg) -> list:
        return []

    def _remote_compute_args(self, msg) -> list:
        return []

    async def _run(self, websocket, path):
        message = await websocket.recv()
        try:
            if message is not None:
                message = _json.loads(message)

        except Exception as e:
            await websocket.close(1011, 'JSON data parse failed')

        if message['mode'] == 'remote':
            try:
                start = _time.time()
                output = await _asyncio.get_event_loop().run_in_executor(
                    None,
                    self._remote(message),
                    *self._remote_compute_args(message)
                )
                _duration(self.cache, start, 'remote_iter_duration')
                info(f"Remote Iter: {self.cache.get('remote_iter_duration', ['undef'])[-1]}", self.debug)
                await websocket.send(_json.dumps({'type': 'stdout', 'data': output, 'end': True}))

            except Exception as e:
                error(e)
                error('Remote data:')
                error(message['data'])
                await websocket.send(_json.dumps({'type': 'stderr', 'data': e, 'end': True}))

        elif message['mode'] == 'local':
            try:
                start = _time.time()
                output = await _asyncio.get_event_loop().run_in_executor(
                    None,
                    self._local(message),
                    *self._local_compute_args(message)
                )
                _duration(self.cache, start, 'local_iter_duration')
                info(f"Local Iter: {self.cache.get('local_iter_duration', ['undef'])[-1]}", self.debug)
                await websocket.send(_json.dumps({'type': 'stdout', 'data': output, 'end': True}))

            except Exception as e:
                error(e)
                error('Local data:')
                error(message['data'])
                await websocket.send(_json.dumps({'type': 'stderr', 'data': e, 'end': True}))
        else:
            await websocket.close()

    def start(self):
        start_server = _ws.serve(self._run, '0.0.0.0', 8881)
        info("Python microservice started on 8881")
        _asyncio.get_event_loop().run_until_complete(start_server)
        _asyncio.get_event_loop().run_forever()
