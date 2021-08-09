#!/usr/bin/env python

"""
Forked from https://pypi.org/project/coinstac/
"""

import asyncio as _asyncio
import json as _json
import time as _time

import websockets as _ws

from coinstac_dinunet.utils import duration as _duration
from coinstac_dinunet.utils.logger import *


class COINPyService:
    def __init__(self, **kw):
        self.cache = kw.get('cache', {})
        self.verbose = kw.get('verbose', False)
        self.profile = kw.get('profile', False)

    def get_local(self, msg) -> callable:
        return ...

    def get_remote(self, msg) -> callable:
        return ...

    def get_local_compute_args(self, msg) -> list:
        return []

    def get_remote_compute_args(self, msg) -> list:
        return []

    async def _run(self, websocket, path):
        message = await websocket.recv()
        try:
            message = _json.loads(message)
        except Exception as e:
            await websocket.close(1011, 'JSON data parse failed')

        if message['mode'] == 'remote':
            try:
                if self.verbose:
                    info(
                        f"ITERATION-{message['data']['state']['iteration']}"
                        f"---------------------------------------------------------------------------------------"
                    )
                    info(f"[*** REMOTE input ***] : {message['data']['input']}")

                start = _time.time()
                output = await _asyncio.get_event_loop().run_in_executor(
                    None,
                    self.get_remote(message),
                    *self.get_remote_compute_args(message)
                )

                if self.profile:
                    _duration(self.cache, start, 'remote_iter_duration')
                    info(f"Remote Iter: {self.cache.get('remote_iter_duration', ['undef'])[-1]}", self.verbose)

                if self.verbose:
                    info(f"[*** REMOTE cache ***]: {self.cache}")
                    info(f"[*** REMOTE output ***]: {output}")

                await websocket.send(_json.dumps({'type': 'stdout', 'data': output, 'end': True}))

            except Exception as e:
                error(e)
                error('Remote data:')
                error(message['data'])
                await websocket.send(_json.dumps({'type': 'stderr', 'data': e, 'end': True}))

        elif message['mode'] == 'local':
            try:
                if self.verbose:
                    info(
                        f"ITERATION-{message['data']['state']['iteration']}"
                        f"---------------------------------------------------------------------------------------"
                    )
                    info(f"[*** {message['data']['state']['clientId']} input ***]: {message['data']['input']}")

                start = _time.time()
                output = await _asyncio.get_event_loop().run_in_executor(
                    None,
                    self.get_local(message),
                    *self.get_local_compute_args(message)
                )

                if self.profile:
                    _duration(self.cache, start, 'local_iter_duration')
                    info(f"Local Iter: {self.cache.get('local_iter_duration', ['undef'])[-1]}", self.verbose)

                if self.verbose:
                    info(f"[*** {message['data']['state']['clientId']} cache ***]: {self.cache}")
                    info(f"[*** {message['data']['state']['clientId']} output ***]: {output}")

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
        success("Python microservice started on 8881")
        _asyncio.get_event_loop().run_until_complete(start_server)
        _asyncio.get_event_loop().run_forever()
