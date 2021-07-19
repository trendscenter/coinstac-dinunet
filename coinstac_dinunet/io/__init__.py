#!/usr/bin/env python

"""
Forked from https://pypi.org/project/coinstac/
"""

import asyncio as _asyncio
import websockets as _ws
import json as _json
from datetime import datetime as _dt


class COINPyService:
    def __init__(self, **kw):
        self.parsed = None

    def _local(self) -> callable:
        return ...

    def _remote(self) -> callable:
        return ...

    def _local_compute_args(self) -> list:
        return []

    def _remote_compute_args(self) -> list:
        return []

    async def _run(self, websocket, path):
        message = await websocket.recv()
        try:
            if message is not None:
                self.parsed = _json.loads(message)

        except Exception as e:
            await websocket.close(1011, 'JSON data parse failed')

        if self.parsed['mode'] == 'remote':
            try:
                start = _dt.now()
                output = await _asyncio.get_event_loop().run_in_executor(
                    None,
                    self._remote(),
                    *self._remote_compute_args()
                )
                print('Remote exec time:')
                print((_dt.now() - start).total_seconds())
                await websocket.send(_json.dumps({'type': 'stdout', 'data': output, 'end': True}))

            except Exception as e:
                print(e)
                print('Remote data:')
                print(self.parsed['data'])
                await websocket.send(_json.dumps({'type': 'stderr', 'data': e, 'end': True}))

        elif self.parsed['mode'] == 'local':
            try:
                start = _dt.now()
                output = await _asyncio.get_event_loop().run_in_executor(
                    None,
                    self._local(),
                    *self._local_compute_args()
                )

                print('Local exec time:')
                print((_dt.now() - start).total_seconds())
                await websocket.send(_json.dumps({'type': 'stdout', 'data': output, 'end': True}))

            except Exception as e:
                print(e)
                print('Local data:')
                print(self.parsed['data'])
                await websocket.send(_json.dumps({'type': 'stderr', 'data': e, 'end': True}))
        else:
            await websocket.close()

    def start(self):
        start_server = _ws.serve(self._run, '0.0.0.0', 8881)
        print("Python microservice started on 8881")

        _asyncio.get_event_loop().run_until_complete(start_server)
        _asyncio.get_event_loop().run_forever()
