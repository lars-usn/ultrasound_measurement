"""
Test configuration and acquisition of Picosciope oscilloscope using classes
and functions from ps5000a_ultrasound_wrappers.py
Bare minimum - Change values to test
Reports results of internal class methods

Created on Wed Aug 19 20:24:21 2026

@author: lars-hoff
"""
import matplotlib.pyplot as plt

import ultrasound_utilities as us         # USN ultrasound lab specific
import ps5000a_ultrasound_wrappers as ps  # Interface to Pico c-library


# -- Initialise variables for instrument control --------------------
dso = ps.Picoscope5000A()
trigger = ps.Trigger()

# -- Initialise variables for waveform provcessing ------------------
wfm = us.Waveform()
pulse = us.Pulse()
rf_filter = us.WaveformFilter()
pulse.dt = 1 / ps.DAC_SAMPLERATE

# --- Horizontal ---------------------------------------------------
sampling = ps.Horizontal()

requested_sample_rate = 100e6
sampling.n_samples = 10000
sampling.trigger_position = 10

print('')
print('Horizontal')
print(f'sample_rate: {sampling.sample_rate}, ', end='')
print(f'n_pretrigger: {sampling.n_pretrigger}, ', end='')
print(f'n_posttrigger: {sampling.n_posttrigger}, ', end='')
print(f'start_time: {sampling.start_time}, ', end='')
print(f'end_time: {sampling.end_time}, ', end='')
print('')

# --- Vertical -----------------------------------------------------
N_CHANNELS = 2
channels = [ps.Channel(k) for k in range(N_CHANNELS)]

no = 0
channel = channels[no]
channel.no = no
channel.enabled = True
channel.v_range = 0.1
channel.offset = 0.0
channel.coupling = ps.Channel.Coupling.DC
channel.bwl: bool = False

no = 1
channel = channels[no]
channel.no = no
channel.enabled = True
channel.v_range = 1
channel.offset = 0.0
channel.coupling = ps.Channel.Coupling.DC
channel.bwl: bool = True

print('')
print('Vertical settings')
for channel in channels:
    print(f'Channel no.: {channel.no}, ', end='')
    print(f'name: {channel.name}, ', end='')
    print(f'v_max: {channel.v_max}, ', end='')
    print(f'coupling_code: {channel.coupling_code}')


# --- Trigger ------------------------------------------------------
trigger.source = ps.Trigger.Source.B
# trigger.source = ps.TriggerSource.EXT
# trigger.source = ps.TriggerSource.INTERNAL
trigger.level = 0.5
trigger.direction = ps.Trigger.Direction.FALLING
trigger.delay = 0.0
trigger.autodelay = 0.01

print('')
print('Trigger')
print(f'enabled: {trigger.enabled}, ', end='')
print(f'source channel no: {trigger.source.channel_no}, ', end='')
print(f'Picoscope name: {trigger.source.picoscope_name}, ',  end='')
print(f'Picoscope source: {trigger.source.picoscope_source}, ',  end='')
print(f'Direction code: {trigger.direction.code}')
print('')


# --- Connect to instrument -----------------------------------------
dso = ps.Picoscope5000A()
try:
    dso.open_adc()
    timebase, sample_rate = dso.find_timebase(requested_sample_rate)
    sampling.timebase = timebase
    sampling.dt = 1/sample_rate
    for channel in channels:
        channel.adc_max = dso.adc_max.value

        # print(channel.adc_max)

except ps.PicoSDKCtypesError:
    print('Could not connect to oscilloscope')

print(f'Oscilloscope connected: {dso.connected}')

# --- Communicate with instrument -----------------------------------
if dso.connected:
    try:
        for channel in channels:
            dso.set_vertical(channel)
            dso.set_bwl(channel)

        dso.set_trigger(trigger, channels, sampling)
        dso.configure_acquisition(sampling)

        y = dso.acquire_trace(sampling, channels)

        plt.plot(y)
    except AttributeError as e:
        print("Error communicating with oscilloscope. Closing")
        print(f"{e}")
    finally:
        dso.close_adc()
else:
    print('Oscilloscope not connected. Skips communication')
