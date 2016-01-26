#!/bin/bash
#
# sample lsf bsub to run an interactive job, optionally on a selected host.
#
# pick a host to land on.
host=${1:-tulgb007}

#
# the -Is says you want an interactive session
# the s says you want a terminal session.
#
# shared_int is the "shared interactive queue"
if [ -z $LSB_BATCH_JID ]; then
  set -x
  bsub \
  -Is \
  -n 1 \
  -q test_int \
  -m $host \
  -W 4200 \
  /bin/bash
fi 
#  -q test_int \
#  -q shared_int \
#  -q excl_int \
