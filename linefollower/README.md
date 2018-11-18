# FCL Linefollower

The line follower demonstrates as a simple scenario how
FCL works. The robot has a default steering mechanism
to track the line and this generates the error signal for FCL.

FCL then learns with the help of this error signal to
improve its behaviour. The inputs to FCL are two rows
of sensors in front of the robot.

## Pre-requisites

Ubuntu Linux: xenial or bionic LTS.

QT5 development libraries with openGL

QT5 version of ENKI:
https://github.com/glasgowneuro/enki

## Compilation

`qmake` and `make` to compile it.

## Running the line follower

The line follower has two modes: single run or stats run.
In the single run mode it runs until the squared average of the
error signal is below a certain threshold (SQ_ERROR_THRES).
In the stats run it performs a logarithmic sweep of different
learning rates and counts the simulation steps till success.

## Data logging

There are two log files: `flog.dat` and `llog.dat`. The
data is space separated and every time step has one row.

### flog.dat

This log records the steering actions of the robot:

`amplified_error steering_left steering_right`

### llog.dat

The error signal can be seen as the performance measure
of learning and it slowly decays to zero which is logged here:

`unamplified_error average_error absolute_error`

Use for example `gnuplot` to plot it with:

```
gnuplot> plot "llog.dat" using 3 with lines
```
