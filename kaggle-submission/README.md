# README

## How to run

### Round 1
In 'Round1.ipynb', make sure all the dependencies are correctly installed, and then run all the code in Round1.ipynb to generate the submission-ready output 'round1.csv'.

### Round 2
In 'Round2.ipynb', make sure all the dependencies are correctly installed, and then run all the code in Round2.ipynb to generate the submission-ready output 'round2.csv'.

## Dependency
All the dependencies are can be found in 'Round1.ipynb' and 'Round2.ipynb' import statements.

Specifically, in order to run locally, a package 'fbprophet' and its dependency 'pystan' need to be carefully installed. Please follow the instructions at here:

https://pypi.org/project/fbprophet/

https://github.com/stan-dev/pystan/blob/readthedocs/doc/installation_beginner.rst

Using command 'conda install pystan' and 'conda install fbprophet' is recommended.

## File Descriptions

The main training and prediction output files are directly under this directory, which are Round1.ipynb Round2.ipynb round1.csv, and round2.csv.

data: all the source data that we needed for our program's input.

googledrive: The MAIN directory that stores all our previous attempts.

arima: The files of attempts of arima models. But the major codes are in googledrive directory.

lstm: The files of attempts of lstm models. But the major codes are in googledrive directory.

other: general modules and other files


