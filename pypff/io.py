'''
This module provides methods to reading pff data file, including img16, img8, ph256, ph1024 and hk.pff
'''
import json
import datetime
import numpy as np
from glob import glob
from . import pixelmap

MOBO_DIM = 16
QUABO_DIM = 32

# The metadata loc in the data is hard-coded here.
loc_arr = np.zeros(2, dtype=object)
# metadata loc for ph256
# metadata example
'''
b'{ "quabo_num": 0, "pkt_num":      32280, "pkt_tai":  398, "pkt_nsec": 723300414, "tv_sec": 1690934633, "tv_usec": 720082}\n'
'''
loc_arr[0] = {
    'quabo_num' : [14, 16],
    'pkt_num'   : [28, 39],
    'pkt_tai'   : [51, 56],
    'pkt_nsec'  : [69, 79],
    'tv_sec'    : [90, 101],
    'tv_usec'   : [113, 120]
}
# metadata loc for ph1024, img16 and img8
# metadata example
'''
'{\n   
    "quabo_0": { "pkt_num":      23855, "pkt_tai":  906, "pkt_nsec": 774507484, "tv_sec": 1691048805, "tv_usec": 778782}, \n   
    "quabo_1": { "pkt_num":      16262, "pkt_tai":  906, "pkt_nsec": 774507492, "tv_sec": 1691048805, "tv_usec": 778789}, \n   
    "quabo_2": { "pkt_num":       9069, "pkt_tai":  906, "pkt_nsec": 774507484, "tv_sec": 1691048805, "tv_usec": 778800}, \n   
    "quabo_3": { "pkt_num":       1234, "pkt_tai":  906, "pkt_nsec": 774507484, "tv_sec": 1691048805, "tv_usec": 778804}
    \n}\n'
'''
loc_arr[1] = {
    'quabo_0':
    {
        'pkt_num': [28 ,39],
        'pkt_tai': [51 ,56],
        'pkt_nsec': [69, 79],
        'tv_sec': [90, 101],
        'tv_usec': [113, 120]
    },
    'quabo_1':
    {
        'pkt_num': [150 ,161],
        'pkt_tai': [173 ,178],
        'pkt_nsec': [191, 201],
        'tv_sec': [212, 223],
        'tv_usec': [235, 242]
    },
    'quabo_2':
    {
        'pkt_num': [272 ,283],
        'pkt_tai': [295 ,300],
        'pkt_nsec': [313, 323],
        'tv_sec': [334, 345],
        'tv_usec': [357, 364]
    },
    'quabo_3':
    {
        'pkt_num': [394 ,405],
        'pkt_tai': [417 ,422],
        'pkt_nsec': [435, 445],
        'tv_sec': [456, 467],
        'tv_usec': [479, 486]
    }
}  
md_loc = {
    'ph256': loc_arr[0],
    'ph1024': loc_arr[1],
    'img16': loc_arr[1],
    'img8': loc_arr[1]
}
# generate dict template
#
def _gen_dict_template(d):
    template = {}
    for k in d:
        # chagne TEMP1 to DET_TEMP, and change TEMP1 to FPGA_TEMP
        if k == 'TEMP1':
            k = 'DET_TEMP'
        if k == 'TEMP2':
            k = 'FPGA_TEMP'
        template[k] = []
    return template

class hkpff(object):
    '''
    Description:
        The hkpff class reads hk.pff, and returns a dict, including housekeeping of quabo, wrs, wps and gps
    '''
    def __init__(self,fn='hk.pff'):
        '''
        Description:
            Create a hkpff object based on the filename.
        Input:
            -- fn(str): file name of a hk.pff 
        '''
        self.fn = fn
        self.hk_info = {}
                
    def readhk(self):
        '''
        Description:
            Read hk.pff, and convert the info to a dict.
        Output:
            -- hk_info(dict): a dict contains all of the hk info.
        '''
        with open(self.fn, 'rb') as f:
            hk_lines = f.readlines()
        for hk_str in hk_lines:
            try:
                hk = json.loads(hk_str)
            except:
                continue
            key, = hk.keys()
            # check if the key is already in the hk_info
            if(not key in self.hk_info):
                template = _gen_dict_template(hk[key])
                self.hk_info[key] = template
            for k,v in hk[key].items():
                # chagne TEMP1 to DET_TEMP, and change TEMP1 to FPGA_TEMP
                if k == 'TEMP1':
                    k = 'DET_TEMP'
                if k == 'TEMP2':
                    k = 'FPGA_TEMP'
                try:
                    # if the type of value is int
                    self.hk_info[key][k].append(int(v))
                except:
                    try:
                        # if the type of value is float
                        self.hk_info[key][k].append(float(v))
                    except:
                        self.hk_info[key][k].append(v)
        return self.hk_info


class datapff(object):
    '''
    Description:
        The datapff class reads all kinds of data files, including img16, img8, ph256, ph1024.
    '''

    def __init__(self, fn):
        '''
        Description:
            Read data from a data pff file.
        Input:
            -- fn(str): pff file name.
        '''
        self.fn = fn
        info = self.fn.split('.')
        stringIndex = 0
        if len(info[0]) != 0:
            stringIndex = 0
        else:
            stringIndex = 1

        startdt_str = info[stringIndex].split('_')[1]
        stringIndex += 1
        # It looks like we have two formats of file name
        self.startdt = datetime.datetime.strptime(startdt_str, '%Y-%m-%dT%H:%M:%SZ')
        self.dp = info[stringIndex].split('_')[1]
        stringIndex += 1
        self.bpp = int(info[stringIndex].split('_')[1])
        stringIndex += 1
        self.module = int(info[stringIndex].split('_')[1])
        stringIndex += 1
        self.seqno = int(info[stringIndex].split('_')[1])
        if self.dp == 'ph256':
            self._md_size = 124
            self._pixels = 256
            self._d_size = self._pixels * self.bpp
            self.datasize = self._md_size + self._d_size
        else:
            # TODO: check if the metadata size for ph1024/img16/img8 is the same
            self._md_size = 492
            self._pixels = 1024
            self._d_size = self._pixels * self.bpp
            self.datasize = self._md_size + self._d_size
        if self.dp == 'ph256' or self.dp == 'ph1024':
            self.dtype = np.int16
        elif self.dp == 'img16':
            self.dtype = np.uint16
        else:
            self.dtype = np.uint8
        self.metadata = {}

    def readpff(self, samples=-1, skip = 0, pixel = -1, ver='qfb', metadata=False):
        '''
        Description:
            Read data from a data pff file.
        Inputs:
            -- samples(int): The sample number to be read out.
                             If it's -1, all of the data will be read out.
                             Default = -1
            -- skip(int): Skip the number of smaples.
                          Default = 0
            -- pixel(int): select the pixel.
                          If it's -1, we will get the data of all the channels.
                          Default = -1
            -- quabo(int): It specifies the quabo number on the mobo.
                          Default = 0
            -- ver(str): quabo version.
                        Default = 'qfp'
        Outputs:
            -- metadata(dict): a dict contains the metadata from each sample.
            -- data(np.array): data array.
        '''
        # get metadata location, which is hard-coded above
        metadata_loc = md_loc[self.dp]
        # read data out from a ph256, img16 or ph1024 file
        with open(self.fn,'rb') as f:
            if samples == -1:
                tmp = np.frombuffer(f.read(),dtype = self.dtype)
            else:
                tmp = np.frombuffer(f.read(samples*self.datasize/self.bpp), dtype=self.dtype)
        # reshape the data
        tmp.shape = (-1, int(self.datasize/self.bpp))
        # get data
        self.data = tmp[:, int(self._md_size/self.bpp):]
        if metadata==True and tmp.shape[0] != 0:
            # we need to skip the '* '
            metadataraw = tmp[:,0: int(self._md_size/self.bpp) - 1]
            metadataraw = metadataraw.tobytes()
            # convert byte to int8
            metadataraw = np.frombuffer(metadataraw, dtype=np.int8)
            metadataraw.shape = (-1, self._md_size - 2)
            # create metadata template
            md_json = json.loads(metadataraw[0].tobytes().decode('utf-8')) 
            if self.dp == 'ph1024' or self.dp == 'img16' or self.dp == 'img8':
                # ph1024, img16 and img8 data has two stages of metadata
                template = _gen_dict_template(md_json)
                for key in template.keys():
                    subtemplate = _gen_dict_template(md_json[key])
                    template[key] = subtemplate
                self.metadata = template
                for k in metadata_loc.keys():
                    for subk in metadata_loc[k].keys():
                        # get the start row and end row from the metadata_loc
                        r0 = metadata_loc[k][subk][0]
                        r1 = metadata_loc[k][subk][1]
                        tmp = metadataraw[:, r0:r1]
                        # covert int8 to string
                        tmp = tmp.view(f'S{r1-r0}')
                        self.metadata[k][subk] = tmp.astype(np.uint64)
            elif self.dp == 'ph256':
                template = _gen_dict_template(md_json)
                # ph256 data has one stage of metadata
                self.metadata = template
                for k in metadata_loc.keys():
                    # get the start row and end row from the metadata_loc
                    r0 = metadata_loc[k][0]
                    r1 = metadata_loc[k][1]
                    tmp = metadataraw[:, r0:r1]
                    # covert int8 to string
                    tmp = tmp.view(f'S{r1-r0}')
                    self.metadata[k] = tmp.astype(np.uint64)
            else:
                raise Exception('Data type is not supproted: %s'%(self.dp))
        
        if pixel != -1:
            self.data = self.data[:,pixel]
        return self.data, self.metadata



class qconfig(object):
    '''
    Description:
        This class is used for reading config json files, including obs_config, daq_config, data_config, quabo_config...
    '''
    def __init__(self, fn):
            self.config = {}
            jfiles = glob(fn)
            if len(jfiles) == 0:
                raise Exception("The config file(%s) can not be found!"%(fn))
            for file in jfiles:
                key = file.split('/')[-1][:-5]
                with open(file,'rb') as f:
                    config = json.load(f)
                self.config[key] = {}
                for k, v in config.items():
                    # if it's quabo_config*, we need to convert the str to int
                    if(key.startswith('quabo_config')):
                        try:
                            tmp = v.split(',')
                        except:
                            tmp = []
                        if len(tmp) == 4:
                            self.config[key][k] = []
                            for vv in tmp:
                                if(vv.startswith('0x')):
                                    self.config[key][k].append(int(vv,16))
                                else:
                                    self.config[key][k].append(int(vv,10))
                        else:
                            if(v.startswith('0x')):
                                self.config[key][k] = int(v,16)
                            else:
                                self.config[key][k] = int(v,10)
                    else:
                        self.config[key] = config