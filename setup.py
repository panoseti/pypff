from setuptools import setup, Extension
import os, glob, numpy, sys 

sys.path.append('pypff')

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '') 
            paths.append(os.path.join(path, filename))
    return paths

data_files = ['pypff']

setup(name='pypff',
    version = '0.0.6',
    description = 'Software for reading pff files generated in PANOSETI project',
    long_description = 'Software for reading pff files generated in PANOSETI project',
    license = 'GPL',
    author = 'Wei Liu',
    author_email = 'liuwei_berkeley@berkeley.edu',
    url = 'http://github.com/liuweiseu/pypff',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],

    install_requires = [
        'numpy',
        'astropy',
        'datetime',
        'pyjson',
        'scipy',
        'matplotlib'
    ],

    package_dir = {'pypff':'pypff'},
    packages = ['pypff'],
    scripts = glob.glob('scripts/*'),
    package_data = {'pypff': data_files},

    include_package_data = True,
)