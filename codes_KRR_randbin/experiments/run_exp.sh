#!/bin/tcsh

set GDB         = gdb
set GDB         = 
set WORKING_DIR = ${HOME}/MyHomeScratch/Research/MachineLearning/codes_KRR_randbin/
set DATA_DIR    = ${HOME}/MyHomeScratch/Research/MachineLearning/data/
set MAXIT       = 50
set TOL         = 1e-3
set VERBOSE     = 1
set NUM_THREADS = 16 
set SIGMA       = (0.01 0.03 0.10 0.14 0.19 0.28 0.39 0.56 0.79 1.12 1.58 2.23 3.16 4.46 6.30 8.91 10 31.62 100)
#set LAMBDA      = (10 1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 0)
set LAMBDA      = (0.01)
set DATASET     = cpu 
#cpu cadata ijcnn1 cod_rna covtype YearPred SUSY pendigits mnist acoustic letter covtype-mult timit

#----------------------------------------------------------------
# Variables specific to each data set

if ( ${DATASET} == cpu ) then

set TRAIN_FILE        = ${DATA_DIR}/cpu/cpu.train
set TEST_FILE         = ${DATA_DIR}/cpu/cpu.test
set NUM_CLASSES       = 1
set DIMENSION         = 12
set RANK              = 8
#8 16 32 64 128 256 512 1024

else if ( ${DATASET} == cadata ) then

set TRAIN_FILE        = ${DATA_DIR}/cadata/cadata.train
set TEST_FILE         = ${DATA_DIR}/cadata/cadata.test
set NUM_CLASSES       = 1
set DIMENSION         = 8
set RANK              = 16
#set RANK              = 258
#2 4 8 16 32 64 129 258 516 1032 2064 4128 8256 16512

else if ( ${DATASET} == ijcnn1 ) then

set TRAIN_FILE        = ${DATA_DIR}/ijcnn1/ijcnn1.train
set TEST_FILE         = ${DATA_DIR}/ijcnn1/ijcnn1.test
set NUM_CLASSES       = 2
set DIMENSION         = 22
set RANK              = 17
#set RANK              = 273
#2 4 8 17 34 68 136 273 546 1093 2187 4375 8750 17500 35000

else if ( ${DATASET} == cod_rna ) then

set TRAIN_FILE        = ${DATA_DIR}/cod-rna/cod_rna.train
set TEST_FILE         = ${DATA_DIR}/cod-rna/cod_rna.test
set NUM_CLASSES       = 2
set DIMENSION         = 8
set RANK              = 12
#set RANK              = 193
#3 6 12 24 48 96 193 386 772 1544 3089 6179 12359 24718 49437

else if ( ${DATASET} == covtype ) then

set SIGMA             = (0.01 0.031623 0.1 0.31623 0.5 1 3.1623)
set LAMBDA            = (1e-4) 
set TRAIN_FILE        = ${DATA_DIR}/covtype/covtype_freq_10.train
set TEST_FILE         = ${DATA_DIR}/covtype/covtype.test
set NUM_CLASSES       = 2
set DIMENSION         = 54
set RANK              = 10
#3 7 14 28 56 113 226 453 907 1815 3631 7262 14525 29050 58101 116202 232404 464809

else if ( ${DATASET} == YearPred ) then

set TRAIN_FILE        = ${DATA_DIR}/YearPred/YearPred.train
set TEST_FILE         = ${DATA_DIR}/YearPred/YearPred.test
set NUM_CLASSES       = 1
set DIMENSION         = 90
set RANK              = 14
#set RANK              = 226
#3 7 14 28 56 113 226 452 905 1810 3621 7242 14484 28969 57939 115879 231759 463518

else if ( ${DATASET} == SUSY ) then

set TRAIN_FILE        = ${DATA_DIR}/SUSY/SUSY.train
set TEST_FILE         = ${DATA_DIR}/SUSY/SUSY.test
set NUM_CLASSES       = 2
set DIMENSION         = 18
set RANK              = 15
#set RANK              = 244
#3 7 15 30 61 122 244 488 976 1953 3906 7812 15625 31250 62500 125000 250000 500000 1000000 2000000 4000000

else if ( ${DATASET} == pendigits ) then

set TRAIN_FILE        = ${DATA_DIR}/pendigits/pendigits.train
set TEST_FILE         = ${DATA_DIR}/pendigits/pendigits.test
set NUM_CLASSES       = 10
set DIMENSION         = 16
#set RANK              = 14
set RANK              = 234
#3 7 14 29 58 117 234 468 936 1873 3747 7494 

else if ( ${DATASET} == mnist ) then

set TRAIN_FILE        = ${DATA_DIR}/mnist/mnist.train
set TEST_FILE         = ${DATA_DIR}/mnist/mnist.test
set NUM_CLASSES       = 10
set DIMENSION         = 780
#set RANK              = 14
#set RANK              = 234
set RANK              = 937
#3 7 14 29 58 117 234 468 937 1875 3750 7500 15000 30000 60000 

else if ( ${DATASET} == acoustic ) then

set SIGMA             = (0.01 0.031623 0.1 0.21623 0.31623 0.41 0.51 0.61 0.71 0.81 0.91 1 3.1623 10 31.623 100)
set LAMBDA            = (1e-1)
set TRAIN_FILE        = ${DATA_DIR}/acoustic/acoustic.train
set TEST_FILE         = ${DATA_DIR}/acoustic/acoustic.test
set NUM_CLASSES       = 3
set DIMENSION         = 50
set RANK              = 10 

else if ( ${DATASET} == letter ) then

set TRAIN_FILE        = ${DATA_DIR}/letter/letter.train
set TEST_FILE         = ${DATA_DIR}/letter/letter.test
set NUM_CLASSES       = 26
set DIMENSION         = 16
set RANK              = 10

else if ( ${DATASET} == covtype-mult ) then

set SIGMA             = (0.01 0.031623 0.1 0.31623 0.5 1 3.1623 5.1623 6.3623 7.5623 10 31.623 100)
#set SIGMA             = (6.3623) # best performance
set LAMBDA            = (1e-4) 
set TRAIN_FILE        = ${DATA_DIR}/covtype-mult/covtype-mult_freq_10.train
set TEST_FILE         = ${DATA_DIR}/covtype-mult/covtype-mult.test
set NUM_CLASSES       = 7
set DIMENSION         = 54
set RANK              = 10 
#8 16 32 64 128 256 512 1024 

else if ( ${DATASET} == timit ) then

set TRAIN_FILE        = ${DATA_DIR}/TIMIT/shuffle/TIMIT_scaled_1E5.train
set TEST_FILE         = ${DATA_DIR}/TIMIT/shuffle/TIMIT_scaled_1E4.test
set NUM_CLASSES       = 147
set DIMENSION         = 440
set RANK              = 1000 
#3 5 9 18 35 69 138 275 550 1100 2199 4398 8796 17591 35181 70362 140724 281447 562893 1125785

endif


#----------------------------------------------------------------
# Set up the output file and final command
set PROG = ${WORKING_DIR}/KRR_OneVsAll_RandBin.ex
set OUT_FILE = RandBin_OneVsAll_${DATASET}_${RANK}.results
#set OUT_FILE = RandBin_${DATASET}_${RANK}_all.results
set COMMAND = "${GDB} ${PROG} ${NUM_THREADS} ${TRAIN_FILE} ${TEST_FILE} ${NUM_CLASSES} ${DIMENSION} ${RANK} $#LAMBDA ${LAMBDA} $#SIGMA ${SIGMA} ${MAXIT} ${TOL} ${VERBOSE}"

#----------------------------------------------------------------
# Run command

rm -f ${OUT_FILE}
echo "DATASET: ${DATASET}" | tee -a ${OUT_FILE}
${COMMAND} | tee -a ${OUT_FILE}

set OUT_FILE_Perf = RandBin_OneVsAll_${DATASET}_${RANK}_Perf.results
grep "RandBinning: OneVsAll." ${OUT_FILE} >> ${OUT_FILE_Perf}
