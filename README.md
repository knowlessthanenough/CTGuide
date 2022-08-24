## Breath Dataset

### Overview

We use temporal convolutional network (TCN) on one breath dataset, described below.

- **breath** dataset (2022) is a lung size dataset of a signal man in age 24 con-
sisting of the entire about 10mins with one 25hz input sensor. In breath dataset, input is a continuous time series having 1 dimensions, output is lung size at the same monent.

The goal here is to predict the 2s later lung size(50 data point later) given 15s (375 data point) sensors history.

### Data

See `data_generator` in `utils.py`. The data has been first pre-processed in to seq to seq data using `time_series_data_slicing` in `utils.py`.


### Note

- Each sequence have the same length. In the current implementation, I train by batch (batch size is 4).

- One can use different datasets by specifying through the `--data` flag on the command line. The
default is `Aligned_Array`.

- The fact that there are 1 dimensions (for 1 feature) .


