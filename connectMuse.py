#!/usr/bin/python2

"""
A module to handle the communication between the Muse and the PC, computing the spectrogram of one channel, and sending the results to a ZMQ client, in real-time.

@author: Hubert Banville
"""

import OSC
import zmq
import threading

import numpy as np
import matplotlib.pyplot as plt

import time
import pickle

import json

from collections import deque


class connectMuse:
    """
    Class that handles:
        1- the communication between Muse-IO and Python using OSC packets OR its simulation using a pre-recorded pickled file
        2- the computation of the spectrogram
        3- the display in matplotlib of the raw EEG and accelerometers signals, and EEG spectrogram
        4- the communication between Python and another script using pyZMQ
        
        Muse-IO has to be started first in a console:
            muse-io --preset 14 --device Muse-6AA1 --osc osc.udp://localhost:4000 --osc-timestamp
        
        Example usage:
        
            plotData = True # Choose between plotting data or just computing the spectrogram
            moc = connectMuseOSC(bufferSize=5, spectrUpdatePeriod=1) # Instantiate the object
            moc.startOSCServer(ipAddress='127.0.0.1', port=4000) # Start reading OSC packets from Muse-IO
            
            if plotData:
                moc.initFigure()
                moc.startPlotTimer() # Start plotting raw EEG, accelerometer signals and EEG spectrogram
            else:
                moc.startSpectrogramComputation(channel=0) # Start the EEG spectrogram for channel 1
                
            moc.startZMQServer() # Start sending spectrogram data over pyZMQ

        
        bufferSize [int]: Length of the buffer for plotting and computing the spectrogram
        ipAddress [string]: IP address for communication with Muse-IO
        port [int]: Port for communication with Muse-IO
        spectrUpdatePeriod [float]: Period at which the spectrogram should be recomputed (and the figure updated)
        
        Dependencies:
            pyOSC (https://pypi.python.org/pypi/pyOSC)
            pyZMQ (https://pypi.python.org/pypi/pyzmq)
            numpy, matplotlib
        Also needs:
            museio (https://sites.google.com/a/interaxon.ca/muse-developer-site/download)
    """
    def __init__(self, bufferSize=10, spectrUpdatePeriod=1):

        self.Fs = 220 # For more robustness, the Fs and accFs should be taken from the OSC packet /muse/config
        self.accFs = 50
        self.bufferSize = bufferSize # seconds
        self.NFFT = 2**self.nextpow2(self.bufferSize) # Next power of 2 from length of y
        
        # Initialize the buffers
        self.eegPackets = deque([[0]*6]*self.Fs*self.bufferSize)
        self.accPackets = deque([[0]*5]*self.accFs*self.bufferSize)
        
        # Initialize the data arrays to plot in real-time
        self.data = np.asarray(self.eegPackets)        
        self.X, self.f, self.t0 = self.stft(self.data[:,0], self.Fs, 1, 0.1)
        self.t = np.arange(0,self.bufferSize,1.0/self.Fs)
        self.accData = np.zeros((self.accFs*self.bufferSize,3))
        
        self.spectrUpdatePeriod = 1 # in seconds
        
    def initOSC(self, ipAddress='127.0.0.1', port=4000):
        """Initialize the OSC connection"""
        
        self.receive_address = (ipAddress, port)
        self.s = OSC.ThreadingOSCServer(self.receive_address)
        
        def format_eeg_handler(addr, tags, stuff, source):
            if addr=='/muse/eeg':
                self.eegPackets.pop()
                self.eegPackets.appendleft(stuff)                
        
        def format_acc_handler(addr, tags, stuff, source):
            if addr=='/muse/acc':
                self.accPackets.pop()
                self.accPackets.appendleft(stuff)                
        
        self.s.addMsgHandler("/muse/eeg", format_eeg_handler)
        self.s.addMsgHandler("/muse/acc", format_acc_handler)
        
    def initFigure(self):
        """Initialize a figure with 3 subplots: Raw EEG signals, EEG spectrogram from channel 1 and accelerometer signals.
            Also initialize a timer that is going to compute the spectrogram and update the plot every [self.spectrUpdatePeriod] seconds."""    
        
        plt.ion()
        self.fig = plt.figure()
        
        # Time signals
        self.ax1 = plt.subplot(311)
        plt.title('EEG data')
        self.ch1, = plt.plot(self.t, self.data[:,0], label='Ch1')
        self.ch2, = plt.plot(self.t, self.data[:,1], label='Ch2')
        self.ch3, = plt.plot(self.t, self.data[:,2], label='Ch3')
        self.ch4, = plt.plot(self.t, self.data[:,3], label='Ch4')
        plt.legend()
        plt.ylim([-400, 400])
        plt.xlim([np.min(self.t), np.max(self.t)])
        
        # Spectrogram
        #Pxx, freqs, bins, im = plt.specgram(np.zeros((self.Fs*self.bufferSize))) #, NFFT=self.NFFT, Fs=self.Fs, noverlap=self.Fs*3)
        plt.subplot(312) #, sharex = self.ax1)       
        self.image = plt.imshow(self.X.T, origin='lower', aspect='auto',
                         interpolation='nearest', extent=[0,self.t0[-1],0,self.f[-1]])
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Spectrogram of channel 1')
        
        # Accelerometer
        plt.subplot(313)
        self.acc1, = plt.plot(self.accData[:,0], label='Forward/Backward')
        self.acc2, = plt.plot(self.accData[:,1], label='Up/Down')
        self.acc3, = plt.plot(self.accData[:,2], label='Left/Right')
        plt.legend()
        plt.ylim([-400, 400])
        plt.title('Accelerometers')
        
        plt.show()
        time.sleep(0.1)
        
        # Create a timer for plotting in real-time
        self.timer = self.fig.canvas.new_timer(interval=self.spectrUpdatePeriod)
        self.timer.add_callback(self.plotSpec)
        
    def initZMQ(self, port=5556):
        """ Initialize the TCP ZMQ connection """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % port)
        
        print('ZMQ connection established.')
        
    def nextpow2(self, i):
        """ Find the next power of 2 for number i """
        n = 1
        while n < i: 
            n *= 2
        return n
        
    def plotSpec(self):
        """Update the figure with 3 subplots"""
        self.data = np.asarray(self.eegPackets)
        self.data = (self.data - np.mean(self.data, axis=0))#/np.std(self.data, axis=0)
        
        self.accData = np.asarray(self.accPackets)
        self.accData = (self.accData - np.mean(self.accData, axis=0))#/np.std(self.accData, axis=0)
        
        # Time signals
        self.ch1.set_ydata(self.data[:,0])
        self.ch2.set_ydata(self.data[:,1])
        self.ch3.set_ydata(self.data[:,2])
        self.ch4.set_ydata(self.data[:,3])
        
        # Spectrogram
        self.X,f,t = self.stft(self.data[:,0], self.Fs, 1, 0.1)
        self.X = np.log10(self.X)
        self.image.set_data(self.X.T)
        self.image.set_clim(np.min(self.X),np.max(self.X)) 
#        plt.subplot(212) #, sharex = ax1)
#        plt.imshow(X.T, origin='lower', aspect='auto',
#                         interpolation='nearest', extent=[0,t[-1],0,f[-1]])
        #Pxx, freqs, bins, im = plt.specgram(data[:,0]) #, NFFT=self.NFFT, Fs=self.Fs) #, noverlap=self.Fs*3)
        
        # Accelerometer
        self.acc1.set_ydata(self.accData[:,0])
        self.acc2.set_ydata(self.accData[:,1])
        self.acc3.set_ydata(self.accData[:,2])
        
        plt.draw()
        
    def processEEG(self, channel):
        """Compute the log spectrogram of the specified raw EEG channel, every [self.spectrUpdatePeriod] seconds."""     
        while True:
            print('*****************')
            self.data = np.asarray(self.eegPackets)
            self.data = (self.data - np.mean(self.data, axis=0))#/np.std(self.data, axis=0)
            
            # Apply spectrogram decomposition
            
            # Pxx is the segments x freqs array of instantaneous power, freqs is
            # the frequency vector, bins are the centers of the time bins in which
            # the power is computed, and im is the matplotlib.image.AxesImage
            # instance
            
            #Pxx, freqs, bins, im = plt.specgram(data[:,0], Fs = self.Fs) #, Fs=Fs, noverlap=900,
            
            self.X,f,t = self.stft(self.data[:,channel], self.Fs, 1, 0.1)
            self.X = np.log10(self.X)
            
            time.sleep(self.spectrUpdatePeriod)
    
    def startOSCServer(self, ipAddress='127.0.0.1', port=4000):
        print 'Initializing the OSC server'
        self.initOSC(ipAddress, port)
        print 'Starting the OSC server'
        st = threading.Thread(target = self.s.serve_forever)
        st.start()
        
    def startPlotTimer(self):
        print 'Starting the plotting timer'
        self.timer.start()

    def startSpectrogramComputation(self, channel):
        print 'Starting the spectrogram computation with channel %i'%(channel+1) 
        pr = threading.Thread(target = self.processEEG, args = [channel])
        pr.start()
        
    def startZMQServer(self, refreshTime, port=5556):
        print 'Initializinfg the ZMQ server'
        self.initZMQ(port)
        print 'Starting the ZMQ server'
        zmqSend = threading.Thread(target = self.sendZMQ, args = [refreshTime])
        zmqSend.start()
        
    def stopOSCServer(self):
        print 'Stopping the OSC server'
        self.s.close()
        
    def pickleData(self):
        """Pickle the raw EEG packets of the plotting buffer."""
        with open('EEG_acc_data.pkl', 'wb') as outputFile:
            pickle.dump((self.eegPackets,self.accPackets), outputFile)
        print('EEG and accelerometer data pickled.')
        
    def unpickleData(self, inputFileName):
        """Unpickle a pre-recorded recording."""
        with open(inputFileName, 'rb') as inputFile:
            print(inputFileName)
            return pickle.load(inputFile)
        
    def sendZMQ(self, refreshTime=1):
        """Send the newest [refreshTime] seconds of spectrogram data via pyZMQ, serialized in json"""
        while True:
            msg2send = self.X[0:self.Fs*refreshTime,:].tolist()
            self.socket.send(json.dumps(msg2send))
            #msg = socket.recv()
            #print msg
            time.sleep(refreshTime)
        
    def stft(self, x, Fs, frameSize, hop):
        """ 
        Performs the Short Time Fourier Transform on a 1D signal x
            Adapted from http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python
            by Steve Tjoa http://stackoverflow.com/users/208339/steve-tjoa
        
        Inputs:
            x: 1D signal (numpy array)
            Fs: Sampling frequency of x
            frameSize: Length of the window on which to apply the FFT, in seconds
            hop: Stride between two consecutive windows, in seconds
            
        Output:
            X: Spectrogram of x (real) [Time x Frequency]
            f: Array of frequencies
            t: Array of time indices (in seconds)
        """
        
        framesamp = int(frameSize*Fs)
        hopsamp = int(hop*Fs)
        w = np.hamming(framesamp)
        X = np.array([abs(np.fft.fft(w*x[i:i+framesamp])) 
                         for i in range(0, len(x)-framesamp, hopsamp)])
        f = np.fft.fftfreq(framesamp, 1./Fs)
        t = np.arange(0, (len(x)-framesamp)/Fs, hop)
        
        # Only keep the part from 0 to Fs/2
        f = f[f>=0]
        X = X[:,0:len(f)]
        
        return X, f, t
        
    def readRecordedEEG(self):
        """Reads one line after the other, every 1/Fs; and then updates the line number; start from the beginning when done"""
        data = self.recEegData[self.eegReadIndex]
        #print data        
        
        self.eegPackets.pop()
        self.eegPackets.appendleft(data)
        if self.eegReadIndex == len(self.recEegData)-1:
            self.eegReadIndex = 0
        else:
            self.eegReadIndex += 1
        threading.Timer(1./self.Fs, self.readRecordedEEG).start()
        
    def readRecordedAcc(self):
        """Reads one line after the other, every 1/accFs; and then updates the line number; start from the beginning when done"""
        data = self.recAccData[self.accReadIndex]
        #print data
        
        self.accPackets.pop()
        self.accPackets.appendleft(data)
        if self.accReadIndex == len(self.recAccData)-1:
            self.accReadIndex = 0
        else:
            self.accReadIndex += 1
        threading.Timer(1./self.accFs, self.readRecordedAcc).start()
    
    def startReadFromFile(self, inputFileName = 'EEG_acc_data.pkl'):
        """Unpickles the recorded dataset, then start the streaming of EEG and Acc data."""
        self.accReadIndex = 0
        self.eegReadIndex = 0
        
        # Unpickle file
        (self.recEegData, self.recAccData) = self.unpickleData(inputFileName)
        
        # Start streaming EEG and Acc data
        self.readRecordedEEG()
        self.readRecordedAcc()
        
if __name__ == "__main__":
    
    plotData = True
    playbackData = True
    
    # Instantiate the connectMuse object
    moc = connectMuse(bufferSize=5)
    
    # Choose where the data comes from: from a pre-recorded pickled file, or from Muse-IO (OSC packets)
    if playbackData:
        moc.startReadFromFile(inputFileName = 'C:\Users\Hubert\Dropbox\Autres\EEG_video_game\OSC_communication_Muse\\EEG_acc_data.pkl')
    else:
        moc.startOSCServer()
    
    # Choose between plotting the signals or only computing the spectrogram
    if plotData:
        moc.initFigure()
        moc.startPlotTimer()
    else:
        moc.startSpectrogramComputation(channel=0)
    
    # Start the ZMQ server for sending the spectrogram data
    moc.startZMQServer(refreshTime=1)


    # When recording...
#    time.sleep(122)
#    moc.stopOSCServer()
#    moc.pickleData()
    
# TODO:
    # Only compute the new part of the spectrogram, not the whole window!
    # Add battery, signal quality information?
    # Directly interface with the Muse protocol instead of using Muse-IO