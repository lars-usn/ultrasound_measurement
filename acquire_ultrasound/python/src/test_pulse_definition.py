# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:36:27 2026

@author: lah
"""
import ultrasound_utilities as us         # USN ultrasound lab specific
import matplotlib.pyplot as plt

pulse = us.Pulse()
pulse.n_cycles = 4
pulse.envelope = us.Window.RECT
pulse.shape = us.Carrier.SAWTOOTH

plt.plot(pulse.y)
