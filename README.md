# Software for PANOSETI project
This python package is used for reading data files genetated by [PANOSETI obs software](https://github.com/panoseti/panoseti), including pff files, config files and housekeeping files.  
## Getting start
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)(optional)  
Miniconda is recommended to create a vritual python environment, so that it won't mess up the python environment on your system.  
If miniconda is installed, please create and activate the python environment.
    ```
    conda create -n wu_env python=3.9
    conda activate wu_env
    ``` 
2. clone the repository
    ```
    git clone https://github.com/liuweiseu/pypff.git
    ```
3. install numpy
    ```
    pip install numpy
    ```
   **Note**: This is because numpy can't be installed automatically, even though it's in the install_requires list.
4. install the package
    ```
    cd pypff
    pip install .
    ```
    **Note:** The packeage is tested under python3.9.16. It should work under python3.x. 

# Introduction
Currently, we only have one module(io.py) in this package, which contains three class: hkpff, datapff and qconfig.
* hkpff: This class is used for reading housekeeping data file.(By default, it's `hk.pff`.)  
    * readhk(): Read hk data from housekeeping data file, and return a dict.
* datapff: This class is used for reading data out from data pff files, including `ph256`, `ph1024`, `img16` and `img8` data files.  
    * readpff: Read data from data files, return a data array and a dict. The dict contains the metadata.
* qconfig: This class is used for reading config files, including `obs_config.json`, `daq_config.json`, `data_config.json` and so on.
    * When the obj is created, you can get the a dict including all of the config information.  

To get more information about how to use the package, please see the example below.

# Example
```python
    import pypff
    import os

    os.chdir('./example-data')

    # read hk.pff
    hkpff = pypff.io.hkpff('hk.pff')
    hk_info = hkpff.readhk()
    print(hk_info)

    # read ph256 data file
    ph256_dpff = pypff.io.datapff('start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff')
    # By default, metadata is not read out, which makes it fasterph256_d, ph256_md = ph256_dpff.readpff(ver='qfb', metadata=True)
    ph256_d, ph256_md = ph256_dpff.readpff(metadata=True)
    print(ph256_md)

    # read img16 data file
    img16_dpff = pypff.io.datapff('start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff')
    # By default, metadata is not read out, which makes it faster.
    img16_d, img16_md = img16_dpff.readpff(metadata=True)
    # check metadata
    print(img16_md)

    # read config files
    # if you don't specify the filename, all the config files will be read
    c = pypff.io.qconfig('*.json')
    print(c.config['obs_config'])
    print(c.config['daq_config'])
    print(c.config['data_config'])
```
Please go to [example](https://github.com/liuweiseu/pypff/tree/master/example) directory to try it.
