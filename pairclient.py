# -*- coding: utf-8 -*-
"""
Receive a json-serialized message.

"""

import zmq
import json

import numpy as np

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

recData = []
recordDuration = 120.
filename = 'C:\Users\Hubert\Dropbox\Autres\EEG_video_game\OSC_communication_Muse\spectrogram1_100ms_256bins.csv'

while True:
    print('Packet received: ')
    msg = json.loads(socket.recv())
    #print msg
    
    recData.append(msg)
    print(len(recData))
    
    if len(recData) > recordDuration/0.1:
        a = np.asarray(recData)
        print a.shape
        np.savetxt(filename, a, delimiter=",")
        print('Spectrogram data of shape '+str(a.shape)+' (recording duration of '+str(recordDuration)+' s) saved in '+filename)        
        
        break