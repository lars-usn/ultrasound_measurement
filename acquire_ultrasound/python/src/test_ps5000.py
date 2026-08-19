"""
Test configuration and acquisition of Picosciope oscilloscope using classes 
and functions from ps5000a_ultrasound_wrappers.py
Bare minimum - Change values to test
Reports results of internal class methods

Created on Wed Aug 19 20:24:21 2026

@author: lars-hoff
"""
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

sampling.timebase = 3
sampling.n_samples = 20000
sampling.dt = 0.8e-9
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
channel = [ps.Channel(k) for k in range(N_CHANNELS)]

no = 0
ch = channel[no]
ch.no = no
ch.enabled = True
ch.v_range = 1.5
ch.adc_max = 32767
ch.offset = 0.0
ch.coupling = ps.Coupling.DC
ch.bwl: bool = False

no = 1
ch = channel[no]
ch.no = no
ch.enabled = True
ch.v_range = 3
ch.adc_max = 32767
ch.offset = 0.0
ch.coupling = ps.Coupling.DC
ch.bwl: bool = False

print('')
print('Vertical settings')
for ch in channel:
    print(f'Channel no.: {ch.no}, ', end='')
    print(f'name: {ch.name}, ', end='')
    print(f'v_max: {ch.v_max}, ', end='')
    print(f'coupling_code: {ch.coupling_code}')


# --- Trigger ------------------------------------------------------
trigger.source = ps.TriggerSource.B
# trigger.source = ps.TriggerSource.EXT
# trigger.source = ps.TriggerSource.INTERNAL
trigger.level = 0.5
trigger.direction = ps.TriggerDirection.FALLING
trigger.delay = 0.0
trigger.autodelay = 0.01
trigger.adc_max = 1

print('')
print('Trigger')
print(f'enabled: {trigger.enabled}, ', end='')
print(f'source channel no: {trigger.source.channel_no}, ', end='')
print(f'Picoscope name: {trigger.source.picoscope_name}, ',  end='')
print(f'Picoscope source: {trigger.source.picoscope_source}, ',  end='')
print(f'Direction code: {trigger.direction.mode}')
print('')


# --- Connect to instrument -----------------------------------------
dso = ps.Picoscope5000A()
try:
    dso.open_adc()
except ps.PicoSDKCtypesError:
    print('Could not connect to oscilloscope')

print(f'Oscilloscope connected: {dso.connected}')

# --- Communicate with instrument -----------------------------------
if dso.connected:
    for ch in channel:
        dso.set_vertical(ch)
        dso.set_bwl(ch)

    dso.set_trigger(trigger, dso.channel, dso.sampling)
    dso.configure_acquisition(sampling)

    y = dso.acquire_trace(sampling, channel)

    dso.close()
else:
    print('Oscilloscope not connected. Skips communication')
