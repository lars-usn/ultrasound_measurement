#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:24:21 2026

@author: lars-hoff
"""
import ultrasound_utilities as us         # USN ultrasound lab specific
import ps5000a_ultrasound_wrappers as ps  # Interface to Pico c-library


# -- Initialise variables for instrument control --------------------
dso = ps.Picoscope5000A()
sampling = ps.Horizontal()
channel = [ps.Channel(i) for i in range(2)]
trigger = ps.Trigger()

# -- Initialise variables for waveform provcessing ------------------
wfm = us.Waveform()
pulse = us.Pulse()
rf_filter = us.WaveformFilter()
pulse.dt = 1 / ps.DAC_SAMPLERATE

# --- Trigger ------------------------------------------------------
trigger.source = ps.TriggerSource.B
# trigger.source = ps.TriggerSource.EXT
# trigger.source = ps.TriggerSource.INTERNAL

trigger.level = 0.5
trigger.direction = ps.TriggerDirection.FALLING
trigger.delay = 0.0
trigger.autodelay = 0.01

trigger.adc_max = 1

print(f'Trigger enabled: {trigger.enabled}')
print(f'Trigger source channel no: {trigger.source.channel_no}')
print(f'Trigger picoscope name: {trigger.source.picoscope_name}')
print(f'Trigger picoscope source: {trigger.source.picoscope_source}')
print(f'Trigger direction code: {trigger.direction.mode}')


# --- Connect to instrument -----------------------------------------
dso = ps.Picoscope5000A()
try:
    dso.open_adc()
except ps.PicoSDKCtypesError:
    print('Could not connect to oscilloscope')

print(f'Oscilloscope connected: {dso.connected}')

# --- Communicate with instrument -----------------------------------
if dso.connected:
    dso.set_trigger(trigger, dso.channel, dso.sampling)

    dso.close()
else:
    print('Oscilloscope not connected. Skips communication')
