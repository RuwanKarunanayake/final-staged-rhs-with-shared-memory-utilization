* Default configuration of execution in GPU set to HYBRID. You can change it from following,
    cuda_gr/include/cudaBSSN.h


If you are executing GPU code with out test(vaerifying the output against CPU) followings are the dependecies,
    bssn/include/def.h
    bssn/include/utils.h
    bssn/src/utils.cpp

Otherwise,
    bssn/include/def.h
    bssn/include/utils.h
    bssn/src/utils.cpp
    bssn/include/rhs.h

* In order to execute output verification against CPU, set the ENABLE_CUDA_TEST compiler flag ON.

* All the files in bssn/cuda_gr/utils are generated by GR/bssnUtils.py