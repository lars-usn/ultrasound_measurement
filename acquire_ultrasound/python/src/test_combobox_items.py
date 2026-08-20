# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:47:46 2026

@author: lah
"""

import ultrasound_utilities as us         # USN ultrasound lab specific
import ps5000a_ultrasound_wrappers as ps  # Interface to Pico c-library


for member in us.Window:
    print(member.value.label, '-', member.value.func_name,
          '-', member.value.par, '-', member)

print()
for member in us.Carrier:
    print(member.value.label, '-', member.value.func_name,
          '-', member.value.par, '-', member)

print()
for member in us.FilterType:
    print(member.value, '-', member.name)

print()
for member in ps.Channel.Coupling:
    print(member.value, '-', member.name)

print()
for member in ps.Trigger.Direction:
    print(member.value, '-', member.name)

print()
for member in ps.Trigger.Source:
    print(member.value, '-', member.name)
