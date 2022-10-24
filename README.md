# Craniosynostosis Distance Maps (CD-Map)

## Overview

This is a Python toolbox to create distance maps, e.g. for a CNN-based 
classification of head deformities. 

We recommend using a virtual environment to manage package dependencies. This
repository can be combined with the instances of the statistical shape model
available at [Zenodo](https://zenodo.org/record/6390158).

## Requirements and installation

The toolbox only depends on Python and should therefore be cross-platform.  
Only some dependencies need to be installed, preferably in a virtual 
environment.

### Setting up the virtual environment

The distance maps toolbox requires `python3`. We recommend using a virtual
environment to install the additional dependencies. Since Python 3.5, the use
of `venv` is recommended: For setting up the environment on any platform, we 
refer to the [official 
documentation](https://docs.python.org/3/library/venv.html). Alternatives to 
the native package management include for example conda. 

On Ubuntu, you can use for example: 

``` bash
python3 -m venv $HOME/venv/cdmap
```

To activate the virtual environment, use:

``` bash
source $HOME/venv/cdmap/bin/activate
```

### Installing the python packages

For installing the python packages we recommend pip which usually ships with
Python by default. Install the following dependencies inside your virtual
environment:

``` bash
pip install numpy scikit-image vtk
```

Alternatively, we provide a requirements file which can also be installed
using pip. However, this will install more packages than required.

We used Python version 3.7, but any reasonably recent version of Python3 should 
be fine.

## Demo

Inside your virtual environment, run:

``` bash
python3 demo.py
```

This will create the distance maps for the four mean shapes from the 
statistical shape model and should be enough to get you going. 

## License
All source code is subject to the terms of the General Public License 3.0.

## References
If you use our code, please cite the corresponding paper:


For a specific release, please also cite the Zenodo release as well:
