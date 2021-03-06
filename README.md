# IRCS
Command-line implementation of basic data reduction of near-infrared images with polarimetry.
The data is from the infrared camera and spectrograph (IRCS) instrument on board the 8-m Subaru telescope in Hawaii.

## Progress
* 2017/03/20: basic scripting
* 2017/04/24: created setup.py
* 2017/04/26: added crop and bgsub
* 2017/04/28: added calflat.py

## Installation
This assumes that you have an environment with at least python 2.7 installed.
If not, use conda to create an environment called `ircs_pol`:
```shell
conda create -n ircs_pol python-2.7 matplotlib numpy
source activate ircs_pol```


Now, clone and then install `ircs`  inside the environment:

1. Clone this repository
```shell
git clone git@github.com:jpdeleon/ircs.git```


2. `cd` into the proper directory and install
```shell
cd /ircs
python setup.py install```


## Sample run

### Part 1 ircs-imaging
```shell
$ ircs-imaging -h```


### Part 2 ircs-polarimetry
```shell
$ ircs-polarimetry -h```


See also other plotting helper functions in /ircs/utils.py.

TO DO: 
1. define input/output directories using a .yaml
2. implement low-level control
3. upgrade to classes
