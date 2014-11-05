# -*- coding: utf-8 -*-
"""
Receive a json-serialized message.

"""

import zmq
import json

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

while True:
    print('Packet received: ')
    msg = json.loads(socket.recv())
    print msg
