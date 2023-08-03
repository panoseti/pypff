'''
This module provides methods to reading pff data file, including img16, img8, ph256, ph1024 and hk.pff
'''
import json
import datetime
import numpy as np

# generate dict template
def _gen_dict_template(d):
    template = {}
    for k in d:
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
        startdt_str = info[0].split('_')[1]
        self.startdt = datetime.datetime.strptime(startdt_str, '%Y-%m-%dT%H:%M:%SZ')
        self.dp = info[1].split('_')[1]
        self.bpp = int(info[2].split('_')[1])
        self.module = int(info[3].split('_')[1])
        self.seqno = int(info[4].split('_')[1])
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
            self.dtpye = np.uint16
        else:
            self.dtype = np.uint8
        self.metadata = {}

    def readpff(self, samples=-1, skip = 0, pixel = -1):
        '''
        Description:
            Read data from a data pff file.
        Inputs:
            -- samples(int): The sample number to be read out.
                             If it's -1, all of the data will be read out.
                             Default = -1
            -- skip(int): Skip the number of smaples.
                          Default = 0
            -- pixel(int): select the channel.
                          If it's -1, we will get the data of all the channels.
                          Default = -1
        Outputs:
            -- metadata(dict): a dict contains the metadata from each sample.
            -- data(np.array): data array.
        '''
        
        f = open(self.fn,'rb')
        # read data out from a ph256 file
        if self.dp == 'ph256':
            if samples == -1:
                tmp = np.frombuffer(f.read(),dtype = self.dtype)
            else:
                tmp = np.frombuffer(f.read(samples*self.datasize/self.bpp), dtype=self.dtype)
            tmp.shape = (-1, int(self.datasize/self.bpp))
            # get data
            self.data = tmp[:, int(self._md_size/self.bpp):]
            # we need to skip the '*'
            metadataraw = tmp[:,0: int(self._md_size/self.bpp) - 1]
            metadataraw = metadataraw.tobytes()
            for md_raw in metadataraw.splitlines():
                md = json.loads(md_raw.decode('utf-8'))
                if not bool(self.metadata):
                    template = _gen_dict_template(md)
                    self.metadata = template
                for k,v in md.items():
                    self.metadata[k].append(v)
        f.close()
        if channel != -1:
            self.data = self.data[:,pixel]
        return self.data, self.metadata
        