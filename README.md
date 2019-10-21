# Using Numba to Speedup Numpy
## Analyzing Chicago Taxi Trips Part 2

This repository contains the slides and code for the talk that I gave at the 
PyLadies X ChiPy event in Chicago on October 16th, 2019.

### How-To Run 

The version that is currently deployed on Binder provides a way to interface
with the knn and dataFormatter objects, but cannot run due to not being able to
read the dataset in, the dataset (in parquet) will be added soon to this repo.

You can also run this locally by clong and installing the env, then running 
a juputer notebook from the source directory. 

### Abstract 

This talk mainly focuses on the fact that using numba can be _incredibly_ useful
to mathmatical operations -- especailly those used in machine learning. This
talk looks at the Chicago Taxi Data Set (Roughly 47 GB when Parquet formatted)
and sees if it can implement a _very_ rudenemntary kNN. Since this is a talk,
this was focused on showing Numba - the odd data science/software engineering
choices were intentionally made in order to show how things are working
together. 

### References

While the code that I wrote is mostly my own, I used some excellent resources
along the way to help me
