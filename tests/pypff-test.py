#! /usr/bin/env python3
import pypff
import os
from glob import glob
# test hk data reading
hkpff = pypff.io.hkpff('hk-data/hk.pff')
hk_info = hkpff.readhk()
# if the files is read out successfully, we should be able to get the keys
print(hk_info.keys())

# test config files reading
c = pypff.io.qconfig('config-data/*.json')
print(c.config['obs_config'].keys())
print(c.config['daq_config'].keys())
print(c.config['data_config'].keys())
print(c.config['network_config'].keys())

# test sci data reading
files = glob('sci-data/*.pff')
for f in files:
    pff = pypff.io.datapff(f)
    data, metadata = pff.readpff(metadata=True)
    print(metadata.keys())