### HDF speed test

To reproduce use:
```commandline
python create_datasets.py 200 0 0
```
this will create hdf file with 200 examples without compression, to add compression change last parameter to `1`.
Next run `main.py` script to benchmark iteration speed.
```commandline
python main.py
```