import pypff
import os

os.chdir('./example-data')

# read hk.pff
hkpff = pypff.io.hkpff('hk.pff')
hk_info = hkpff.readhk()
print(hk_info)

# read ph256 data file
ph256_dpff = pypff.io.datapff('start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff')
# By default, metadata is not read out, which makes it faster.
ph256_d, ph256_md = ph256_dpff.readpff(ver='qfb', metadata=True)
# check metadata
print(ph256_md)

# read img16 data file
img16_dpff = pypff.io.datapff('start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff')
# By default, metadata is not read out, which makes it faster.
img16_d, img16_md = img16_dpff.readpff(ver='qfb', metadata=True)
# check metadata
print(img16_md)

# read config files
# if you don't specify the filename, all the config files will be read
c = pypff.io.qconfig('*.json')
print(c.config['obs_config'])
print(c.config['daq_config'])
print(c.config['data_config'])