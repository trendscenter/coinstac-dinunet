#!/usr/bin/env python

"""
Forked from https://pypi.org/project/coinstac/
"""

import asyncio as _asyncio
import json as _json
import multiprocessing as _mp
import time as _time

import websockets as _ws

from coinstac_dinunet.utils import duration as _duration
from coinstac_dinunet.utils.logger import *


class COINPyService:
    def __init__(self, **kw):
        self.cache = kw.get('cache', {})
        self.verbose = kw.get('verbose', False)
        self.profile = kw.get('profile', False)
        self.mp_pool = None

    def get_local(self, msg):
        r"""Should return a callable module and arguments to the compute function of that module"""
        return ...

    def get_remote(self, msg):
        r"""Should return a callable module and arguments to the compute function of that module"""
        return ...

    async def _run(self, websocket, path):
        message = await websocket.recv()
        try:
            message = _json.loads(message)
        except Exception as e:
            await websocket.close(1011, 'JSON data parse failed')

        if self.mp_pool is None:
            self.mp_pool = _mp.Pool(processes=message['data']['input'].get('num_reducers', 2))

        if message['mode'] == 'remote':
            try:
                if self.verbose:
                    info(
                        f"ITERATION-{message['data']['state']['iteration']}"
                        f"---------------------------------------------------------------------------------------"
                    )
                    info(f"[*** REMOTE input ***] : {message['data']['input']}")

                start = _time.time()
                remote, remote_args = self.get_remote(msg=message), [self.mp_pool]
                if isinstance(remote, tuple) or isinstance(remote, list):
                    remote_args = [self.mp_pool] + list(remote)[1:]
                output = await _asyncio.get_event_loop().run_in_executor(None, remote[0], *remote_args)

                if self.profile:
                    _duration(self.cache, start, 'remote_iter_duration')
                    info(f"Remote Iter: {self.cache.get('remote_iter_duration', ['undef'])[-1]}")

                if self.verbose:
                    info(f"[***** REMOTE cache *****]: {self.cache}")
                    info(f"[***** REMOTE output *****]: {output}")
                    info("==========================================================================================")

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
                    info(f"[***** {message['data']['state']['clientId']} input *****]: {message['data']['input']}")

                start = _time.time()
                local, local_args = self.get_local(msg=message), [self.mp_pool]
                if isinstance(local, tuple) or isinstance(local, list):
                    local_args = [self.mp_pool] + list(local)[1:]
                output = await _asyncio.get_event_loop().run_in_executor(None, local[0], *local_args)

                if self.profile:
                    _duration(self.cache, start, 'local_iter_duration')
                    info(f"Local Iter: {self.cache.get('local_iter_duration', ['undef'])[-1]}")

                if self.verbose:
                    info(f"[***** {message['data']['state']['clientId']} cache *****]: {self.cache}")
                    info(f"[***** {message['data']['state']['clientId']} output *****]: {output}")
                    info("==========================================================================================")

                await websocket.send(_json.dumps({'type': 'stdout', 'data': output, 'end': True}))

            except Exception as e:
                error(e)
                error(f'{message["data"]["state"]["clientId"]} data:')
                error(message['data'])
                await websocket.send(_json.dumps({'type': 'stderr', 'data': e, 'end': True}))
        else:
            await websocket.close()

    def start(self):
        start_server = _ws.serve(self._run, '0.0.0.0', 8881)
        success("Python microservice started on 8881")
        _asyncio.get_event_loop().run_until_complete(start_server)
        _asyncio.get_event_loop().run_forever()
