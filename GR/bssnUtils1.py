##########################################################################
# 
# Created on: Sep 23, 2018
#       Author: Akila, Eranga, Eminda, Ruwan
# 
##########################################################################

from collections import namedtuple
from datetime import datetime
from time import strftime

import dendro as dendro
import math as math
import sympy as sympy
import re as re

def addHeader(file, location):
    template = "// Generated by Dendro-GR SymPyGR code generation framework\n// date: {}\n// location: {}\n\n"
    date_time = datetime.now().isoformat(" ")
    file.write(template.format(date_time, location))

## ==== BSSN GPU code generation paramerters

# enum to symbolic input vars dictionary
varEnumToInputSymbol={ "alpha" : "alphaInt",
                       "beta0" : "beta0Int",
                       "beta1" : "beta1Int",
                       "beta2" : "beta2Int",
                       "B0"    : "B0Int",
                       "B1"    : "B1Int",
                       "B2"    : "B2Int",
                       "chi"   : "chiInt",
                       "Gt0"   : "Gt0Int",
                       "Gt1"   : "Gt1Int",
                       "Gt2"   : "Gt2Int",
                       "K"     : "KInt",
                       "gt0"   : "gt0Int",
                       "gt1"   : "gt1Int",
                       "gt2"   : "gt2Int",
                       "gt3"   : "gt3Int",
                       "gt4"   : "gt4Int",
                       "gt5"   : "gt5Int",
                       "At0"   :"At0Int",
                       "At1"   :"At1Int",
                       "At2"   :"At2Int",
                       "At3"   :"At3Int",
                       "At4"   :"At4Int",
                       "At5"   :"At5Int"
                      }

# first derivs required for RHS
d = [
    "alpha", "beta0", "beta1", "beta2",
    "B0", "B1", "B2",
    "chi", "Gt0", "Gt1", "Gt2", "K",
    "gt0", "gt1", "gt2", "gt3", "gt4", "gt5",
    "At0", "At1", "At2", "At3", "At4", "At5" 
    ]

# second derivs required for RHS
dd = [
    "gt0", "gt1", "gt2", "gt3", "gt4", "gt5", "chi",
    "alpha", "beta0", "beta1", "beta2" 
    ]

# advective derivatives
ad = [
    "gt0", "gt1", "gt2", "gt3", "gt4", "gt5",
    "At0", "At1", "At2", "At3", "At4", "At5",
    "alpha", "beta0", "beta1", "beta2", "chi", "Gt0", "Gt1", "Gt2", "K",
    "B0", "B1", "B2"
    ] 

# first derivs required for constraints--no gauge variables
con_d = [ 
    "chi", "Gt0", "Gt1", "Gt2", "K",
    "gt0", "gt1", "gt2", "gt3", "gt4", "gt5",
    "At0", "At1", "At2", "At3", "At4", "At5" 
    ]

# second derivs required for constraints--no gauge variables
con_dd = ["gt0", "gt1", "gt2", "gt3", "gt4", "gt5", "chi"]


pd = ["grad_0_", "grad_1_", "grad_2_"]
pad = ["agrad_0_", "agrad_1_", "agrad_2_"]
pkod = ["kograd_0_", "kograd_1_", "kograd_2_"]
pdd = ["grad2_0_0_", "grad2_0_1_", "grad2_0_2_", "grad2_1_1_", "grad2_1_2_", "grad2_2_2_"]

# RHS derivatives...................................................
funcs=[]
# first derivative in i direction
for f in d:
    for p in pd:
        funcs.append(p+f)

# second derivative in ij direction
for f in dd:
    for p in pdd:
        funcs.append(p+f)

# advective derivatives...................................................
afuncs=[]

#advective derivative in i direction
for f in ad:
    for p in pad:
        afuncs.append(p+f)

#Kriess-Oliger derivative in i direction
kofuncs=[]
for f in d:
    for p in pkod:
        kofuncs.append(p+f)

###########################################################################
#
#  Declare references for derivatives, advective derivatives & BSSN variables
#
###########################################################################
# Declare references for derive and adv_deriv
with open("generated/declare_ref_derivs.h", "w") as declare_deriv_file:
    addHeader(declare_deriv_file, "bssn/cuda_gr/utils")
    
    # double * agrad_0_gt1;
    template = "double * {};\n"
    for var in funcs: declare_deriv_file.write(template.format(var))
    for var in afuncs: declare_deriv_file.write(template.format(var))

# Set offset for the derive and adv_deriv arrays
with open("generated/args_offset_derivs.h", "w") as args_deriv_file:
    addHeader(args_deriv_file, "bssn/cuda_gr/utils")
    
    # &grad_0_alpha[unzip_dof * streamIndex],
    template = "&{}[unzip_dof * streamIndex],\n"
    for var in funcs: args_deriv_file.write(template.format(var))
    for var in afuncs[:-1]: args_deriv_file.write(template.format(var))
    args_deriv_file.write("&{}[unzip_dof * streamIndex]\n".format(afuncs[-1]))

# Args of derive, adv_deriv arrays, bssn variables
with open("generated/args_derivs_offsets.h", "w") as args_deriv_file:
    addHeader(args_deriv_file, "bssn/cuda_gr/utils")
    
    # grad_0_alpha,
    template = "{},\n"
    for offset in d: args_deriv_file.write(template.format(varEnumToInputSymbol[offset]))
    for var in funcs: args_deriv_file.write(template.format(var))
    for var in afuncs[:-1]: args_deriv_file.write(template.format(var))
    args_deriv_file.write("{}\n".format(afuncs[-1]))

# Para of derive, adv_deriv arrays, bssn variables
with open("generated/para_derivs_offsets.h", "w") as para_deriv_file:
    addHeader(para_deriv_file, "bssn/cuda_gr/utils")
    
    # int alphaInt,
    # double *grad_1_At0,
    template_offset = "int {},\n"
    template_deriv = "double * {},\n"
    for offset in d: para_deriv_file.write(template_offset.format(varEnumToInputSymbol[offset]))
    for var in funcs: para_deriv_file.write(template_deriv.format(var))
    for var in afuncs[:-1]: para_deriv_file.write(template_deriv.format(var))
    para_deriv_file.write("double * {}\n".format(afuncs[-1]))

# Para of derive, adv_deriv arrays
with open("generated/para_derivs.h", "w") as para_deriv_file:
    addHeader(para_deriv_file, "bssn/cuda_gr/utils")
    
    # int alphaInt,
    # double *grad_1_At0,
    template_deriv = "double * {},\n"
    for var in funcs: para_deriv_file.write(template_deriv.format(var))
    for var in afuncs[:-1]: para_deriv_file.write(template_deriv.format(var))
    para_deriv_file.write("double * {}\n".format(afuncs[-1]))


###########################################################################
#
#  Allocate memory derivatives & advective derivatives
#
###########################################################################
with open("generated/bssnrhs_cuda_malloc.h", "w") as funcs_alloc_file:
    addHeader(funcs_alloc_file, "bssn/cuda_gr/utils")

    funcs_alloc_file.write("// GPU memory allocation for derivatives\n")

    # CHECK_ERROR(cudaMalloc((void **) &grad_0_alpha, size), "grad_0_alpha");
    template = "CHECK_ERROR(cudaMalloc((void **) &{}, size), \"{} GPU memory allocation failed\");\n"
    for var in funcs: funcs_alloc_file.write(template.format(var, var))

    funcs_alloc_file.write("\n// GPU memory allocation for advective derivatives\n")

    # CHECK_ERROR(cudaMalloc((void **) &agrad_0_gt0, size), "agrad_0_gt0");
    template_adv = "CHECK_ERROR(cudaMalloc((void **) &{}, size), \"{} GPU memory allocation failed\");\n"
    for var in afuncs: funcs_alloc_file.write(template_adv.format(var, var))


###########################################################################
#
# Deallocate memory derivatives & advective derivatives
#
###########################################################################
with open("generated/bssnrhs_cuda_mdealloc.h", "w") as funcs_dealloc_file:
    addHeader(funcs_dealloc_file, "bssn/cuda_gr/utils")

    funcs_dealloc_file.write("// Release GPU memory allocated for derivatives\n")

    # CHECK_ERROR(cudaFree(grad_0_alpha), "grad_0_alpha cudafree");
    template = "CHECK_ERROR(cudaFree({}), \"{} memory releaseing failed\");\n"
    for var in funcs: funcs_dealloc_file.write(template.format(var, var))

    funcs_dealloc_file.write("\n// Release GPU memory allocated for advective derivatives\n")

    # CHECK_ERROR(cudaFree(agrad_1_gt0), "agrad_1_gt0 cudafree");
    template_adv = "CHECK_ERROR(cudaFree({}), \"{} memory releaseing failed\");\n"
    for var in afuncs: funcs_dealloc_file.write(template_adv.format(var, var))


###########################################################################
#
#  Calls for derivatives - 1
#
###########################################################################
with open("generated/calc_deriv_calls_1.cuh", "w") as funcs_call_file:
    addHeader(funcs_call_file, "bssn/cuda_gr/utils")

    helo_length = 2
    # globaltoshared(dev_var_in, shared_var_in, alphaInt, 2, x_offset, y_offset, z_offset, nx, ny);
    template_global_to_shared = "globaltoshared(dev_var_in, shared_var_in, {}, {}, x_offset, y_offset, z_offset, nx, ny);\n"
    # calc_deriv42_x(tile_size, grad_0_alpha, shared_var_in, alphaInt, hx, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_x = "calc_deriv42_x(tile_size, {}, shared_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_xx(tile_size, grad2_0_0_alpha, shared_var_in, alphaInt, hx, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_xx = "calc_deriv42_xx(tile_size, {}, shared_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    # calc_deriv42_y(tile_size, grad_1_alpha, shared_var_in, alphaInt, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_y = "calc_deriv42_y(tile_size, {}, shared_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_yy(tile_size, grad2_1_1_alpha, shared_var_in, alphaInt, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_yy = "calc_deriv42_yy(tile_size, {}, shared_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    # calc_deriv42_z(tile_size, grad_2_alpha, shared_var_in, alphaInt, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_z = "calc_deriv42_z(tile_size, {}, shared_var_in, {}, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_zz(tile_size, grad2_2_2_alpha, shared_var_in, alphaInt, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_zz = "calc_deriv42_zz(tile_size, {}, shared_var_in, {}, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in d:
        funcs_call_file.write(template_global_to_shared.format(varEnumToInputSymbol[offset], helo_length))

        dxn = "grad_0_" + offset
        dxxn = "grad2_0_0_" + offset
        funcs_call_file.write(template_x.format(dxn, varEnumToInputSymbol[offset]))
        if offset in dd: funcs_call_file.write(template_xx.format(dxxn, varEnumToInputSymbol[offset]))

        dyn = "grad_1_" + offset
        dyyn = "grad2_1_1_" + offset
        funcs_call_file.write(template_y.format(dyn, varEnumToInputSymbol[offset]))
        if offset in dd: funcs_call_file.write(template_yy.format(dyyn, varEnumToInputSymbol[offset]))

        dzn = "grad_2_" + offset
        dzzn = "grad2_2_2_" + offset
        funcs_call_file.write(template_z.format(dzn, varEnumToInputSymbol[offset]))
        if offset in dd: funcs_call_file.write(template_zz.format(dzzn, varEnumToInputSymbol[offset]))
        
        funcs_call_file.write("\n")

####  Calls for derivatives - 1 when bflag set ####
with open("generated/calc_deriv_calls_1_bflag.cuh", "w") as funcs_call_file:
    addHeader(funcs_call_file, "bssn/cuda_gr/utils")

    # calc_deriv42_x_bflag(tid, grad_0_alpha, dev_var_in, alphaInt, hx, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_x = "calc_deriv42_x_bflag(tid, {}, dev_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_xx_bflag(tid, grad2_0_0_alpha, dev_var_in, alphaInt, hx, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_xx = "calc_deriv42_xx_bflag(tid, {}, dev_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    # calc_deriv42_y_bflag(tid, grad_1_alpha, dev_var_in, alphaInt, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_y = "calc_deriv42_y_bflag(tid, {}, dev_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_yy_bflag(tid, grad2_1_1_alpha, dev_var_in, alphaInt, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_yy = "calc_deriv42_yy_bflag(tid, {}, dev_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    # calc_deriv42_z_bflag(tid, grad_2_alpha, dev_var_in, alphaInt, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_z = "calc_deriv42_z_bflag(tid, {}, dev_var_in, {}, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_zz_bflag(tid, grad2_2_2_alpha, dev_var_in, alphaInt, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_zz = "calc_deriv42_zz_bflag(tid, {}, dev_var_in, {}, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in d:
        dxn = "grad_0_" + offset
        dxxn = "grad2_0_0_" + offset
        funcs_call_file.write(template_x.format(dxn, varEnumToInputSymbol[offset]))
        if offset in dd: funcs_call_file.write(template_xx.format(dxxn, varEnumToInputSymbol[offset]))

        dyn = "grad_1_" + offset
        dyyn = "grad2_1_1_" + offset
        funcs_call_file.write(template_y.format(dyn, varEnumToInputSymbol[offset]))
        if offset in dd: funcs_call_file.write(template_yy.format(dyyn, varEnumToInputSymbol[offset]))

        dzn = "grad_2_" + offset
        dzzn = "grad2_2_2_" + offset
        funcs_call_file.write(template_z.format(dzn, varEnumToInputSymbol[offset]))
        if offset in dd: funcs_call_file.write(template_zz.format(dzzn, varEnumToInputSymbol[offset]))
        
        funcs_call_file.write("\n")
        
###########################################################################
#
#  Calls for mixed 2nd derivatives & advective derivatives
#
###########################################################################
with open("generated/calc_deriv_calls_2.cuh", "w") as funcs_call_file:
    addHeader(funcs_call_file, "bssn/cuda_gr/utils")

    helo_length = 2
    helo_length_adv = 3
    # globaltoshared(grad_0_gt0, shared_var_in, 0, 2, x_offset, y_offset, z_offset, nx, ny);
    template_global_to_shared = "globaltoshared({}, shared_var_in, 0, {}, x_offset, y_offset, z_offset, nx, ny);\n"
    # calc_deriv42_y(tile_size, grad2_0_1_gt0, grad_0_gt0, 0, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_y = "calc_deriv42_y(tile_size, {}, shared_var_in, 0, hy, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_z(tile_size, grad2_0_2_gt0, grad_0_gt0, 0, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_z = "calc_deriv42_z(tile_size, {}, shared_var_in, 0, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in dd:
        dxn = "grad_0_" + offset
        dyn = "grad_1_" + offset
        dxyn = "grad2_0_1_" + offset
        dxzn = "grad2_0_2_" + offset
        dyzn = "grad2_1_2_" + offset

        funcs_call_file.write(template_global_to_shared.format(dxn, helo_length))
        funcs_call_file.write(template_y.format(dxyn, dxn))
        funcs_call_file.write(template_z.format(dxzn, dxn))

        funcs_call_file.write("\n")

        funcs_call_file.write(template_global_to_shared.format(dyn, helo_length))
        funcs_call_file.write(template_z.format(dyzn, dyn))

        funcs_call_file.write("\n")

    funcs_call_file.write("\n")
    funcs_call_file.write("\n")

    template_global_to_shared_adv = "globaltoshared(dev_var_in, shared_var_in, {}, {}, x_offset, y_offset, z_offset, nx, ny);\n"
    # calc_deriv42_adv_x(tile_size, agrad_0_alpha, dev_var_in, alphaInt, hx, beta0Int, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_adv_x = "calc_deriv42_adv_x(tile_size, {}, dev_var_in, shared_var_in, {}, hx, beta0Int, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_adv_y(tile_size, agrad_1_alpha, dev_var_in, alphaInt, hy, beta1Int, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_adv_y = "calc_deriv42_adv_y(tile_size, {}, dev_var_in, shared_var_in, {}, hy, beta1Int, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_adv_z(tile_size, agrad_2_gt4, dev_var_in, gt4Int, hz, beta2Int, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_adv_z = "calc_deriv42_adv_z(tile_size, {}, dev_var_in, shared_var_in, {}, hz, beta2Int, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in ad:
        dxn = "agrad_0_" + offset
        dyn = "agrad_1_" + offset
        dzn = "agrad_2_" + offset

        funcs_call_file.write(template_global_to_shared_adv.format( varEnumToInputSymbol[offset], helo_length_adv))
        funcs_call_file.write(template_adv_x.format(dxn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_adv_y.format(dyn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_adv_z.format(dzn, varEnumToInputSymbol[offset]))

        funcs_call_file.write("\n")

####  Calls for mixed 2nd derivatives & advective derivatives when bflag set ####
with open("generated/calc_deriv_calls_2_bflag.cuh", "w") as funcs_call_file:
    addHeader(funcs_call_file, "bssn/cuda_gr/utils")

    # calc_deriv42_y_bflag(tid, grad2_0_1_gt0, grad_0_gt0, 0, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_y = "calc_deriv42_y_bflag(tid, {}, {}, 0, hy, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_z_bflag(tid, grad2_0_2_gt0, grad_0_gt0, 0, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_z = "calc_deriv42_z_bflag(tid, {}, {}, 0, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in dd:
        dxn = "grad_0_" + offset
        dyn = "grad_1_" + offset
        dxyn = "grad2_0_1_" + offset
        dxzn = "grad2_0_2_" + offset
        dyzn = "grad2_1_2_" + offset

        funcs_call_file.write(template_y.format(dxyn, dxn))
        funcs_call_file.write(template_z.format(dxzn, dxn))
        funcs_call_file.write(template_z.format(dyzn, dyn))

        funcs_call_file.write("\n")

    # calc_deriv42_adv_x_bflag(tid, agrad_0_alpha, dev_var_in, alphaInt, hx, beta0Int, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_adv_x = "calc_deriv42_adv_x_bflag(tid, {}, dev_var_in, {}, hx, beta0Int, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_adv_y_bflag(tid, agrad_1_alpha, dev_var_in, alphaInt, hy, beta1Int, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_adv_y = "calc_deriv42_adv_y_bflag(tid, {}, dev_var_in, {}, hy, beta1Int, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_deriv42_adv_z_bflag(tid, agrad_2_gt4, dev_var_in, gt4Int, hz, beta2Int, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_adv_z = "calc_deriv42_adv_z_bflag(tid, {}, dev_var_in, {}, hz, beta2Int, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in ad:
        dxn = "agrad_0_" + offset
        dyn = "agrad_1_" + offset
        dzn = "agrad_2_" + offset

        funcs_call_file.write(template_adv_x.format(dxn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_adv_y.format(dyn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_adv_z.format(dzn, varEnumToInputSymbol[offset]))

        funcs_call_file.write("\n")

###########################################################################
#
#  Calls for Kreiss-Oliger derivatives
#  Use the same storage as for the standard first derivatives.
#
###########################################################################
with open("generated/calc_ko_deriv_calls.cuh", "w") as funcs_call_file:
    addHeader(funcs_call_file, "bssn/cuda_gr/utils")

    # calc_ko_deriv42_x(tid, grad_0_gt0, dev_var_in, gt0Int, hx, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_ko_x = "calc_ko_deriv42_x(tid, {}, dev_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_ko_deriv42_y(tid, grad_1_gt0, dev_var_in, gt0Int, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_ko_y = "calc_ko_deriv42_y(tid, {}, dev_var_in, {}, hy, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_ko_deriv42_z(tid, grad_2_gt0, dev_var_in, gt0Int, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_ko_z = "calc_ko_deriv42_z(tid, {}, dev_var_in, {}, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in ad:
        dxn = "grad_0_" + offset
        dyn = "grad_1_" + offset
        dzn = "grad_2_" + offset

        funcs_call_file.write(template_ko_x.format(dxn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_ko_y.format(dyn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_ko_z.format(dzn, varEnumToInputSymbol[offset]))

        funcs_call_file.write("\n")

####  Calls for Kreiss-Oliger derivatives when bflag set ####
with open("generated/calc_ko_deriv_calls_bflag.cuh", "w") as funcs_call_file:
    addHeader(funcs_call_file, "bssn/cuda_gr/utils")

    # calc_ko_deriv42_x_bflag(tid, grad_0_gt0, dev_var_in, gt0Int, hx, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_ko_x = "calc_ko_deriv42_x_bflag(tid, {}, dev_var_in, {}, hx, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_ko_deriv42_y_bflag(tid, grad_1_gt0, dev_var_in, gt0Int, hy, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_ko_y = "calc_ko_deriv42_y_bflag(tid, {}, dev_var_in, {}, hy, host_sz_x, host_sz_y, host_sz_z, bflag);\n"
    # calc_ko_deriv42_z_bflag(tid, grad_2_gt0, dev_var_in, gt0Int, hz, host_sz_x, host_sz_y, host_sz_z, bflag);
    template_ko_z = "calc_ko_deriv42_z_bflag(tid, {}, dev_var_in, {}, hz, host_sz_x, host_sz_y, host_sz_z, bflag);\n"

    for offset in ad:
        dxn = "grad_0_" + offset
        dyn = "grad_1_" + offset
        dzn = "grad_2_" + offset

        funcs_call_file.write(template_ko_x.format(dxn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_ko_y.format(dyn, varEnumToInputSymbol[offset]))
        funcs_call_file.write(template_ko_z.format(dzn, varEnumToInputSymbol[offset]))

        funcs_call_file.write("\n")

###########################################################################
#
#  Generating code for derivs with shared memory
#
###########################################################################

# positions Direction Wise
dirnVisePositions = {"calc_deriv42_x" : [3, 1, 1],
                     "calc_deriv42_y": [3, 3, 1],
                     "calc_deriv42_z": [3, 3, 3],
                     "calc_deriv42_xx": [3, 3, 3],
                     "calc_deriv42_yy": [3, 3, 3],
                    "calc_deriv42_zz" : [3, 3, 3],
                    "calc_deriv42_adv_x" : [3, 3, 3],
                    "calc_deriv42_adv_y" : [3, 3, 3],
                    "calc_deriv42_adv_z" : [3, 3, 3]
                     }

# idx by values
idx_by = {"calc_deriv42_x" : "idx_by_12",
            "calc_deriv42_y": "idy_by_12",
            "calc_deriv42_z": "idz_by_12",
            "calc_deriv42_xx": "idx_sqrd_by_12",
            "calc_deriv42_yy": "idy_sqrd_by_12",
            "calc_deriv42_zz" : "idz_sqrd_by_12",
            "calc_deriv42_adv_x" : "idx_by_12",
            "calc_deriv42_adv_y" : "idy_by_12",
            "calc_deriv42_adv_z" : "idz_by_12"
            }

# deriv wise access locations
stencils = {"calc_deriv42_x" : ["", 1, "idx_by_12"],
            "calc_deriv42_y": ["*16", 16, "idy_by_12"],
            "calc_deriv42_z": ["*16*16", "16*16", "idz_by_12"],
            "calc_deriv42_xx": ["", 1, "idx_sqrd_by_12"],
            "calc_deriv42_yy": ["*16", 16, "idy_sqrd_by_12"],
            "calc_deriv42_zz" : ["*16*16", "16*16", "idz_sqrd_by_12"]
            }

# output line setting
outputLine = {"calc_deriv42_x" : "\toutput[pp] = (dev_var_in[loc_pp - 2{}] - 8.0 * dev_var_in[loc_pp - {}] + "
                                 "8.0 * dev_var_in[loc_pp + {}] - dev_var_in[loc_pp + 2{}]) * {};\n",
            "calc_deriv42_y": "\toutput[pp] = (dev_var_in[loc_pp - 2{}] - 8.0 * dev_var_in[loc_pp - {}] + "
                                 "8.0 * dev_var_in[loc_pp + {}] - dev_var_in[loc_pp + 2{}]) * {};\n",
            "calc_deriv42_z": "\toutput[pp] = (dev_var_in[loc_pp - 2{}] - 8.0 * dev_var_in[loc_pp - {}] + "
                                 "8.0 * dev_var_in[loc_pp + {}] - dev_var_in[loc_pp + 2{}]) * {};\n",
            "calc_deriv42_xx": "\toutput[pp] = ((-1)*dev_var_in[loc_pp-2{}] + 16.0*dev_var_in[loc_pp-{}] "
                               "- 30.0*dev_var_in[loc_pp] + 16.0*dev_var_in[loc_pp+{}] - dev_var_in[loc_pp+2{}])*{};\n",
            "calc_deriv42_yy":  "\toutput[pp] = ((-1)*dev_var_in[loc_pp-2{}] + 16.0*dev_var_in[loc_pp-{}] "
                               "- 30.0*dev_var_in[loc_pp] + 16.0*dev_var_in[loc_pp+{}] - dev_var_in[loc_pp+2{}])*{};\n",
            "calc_deriv42_zz": "\toutput[pp] = ((-1)*dev_var_in[loc_pp-2{}] + 16.0*dev_var_in[loc_pp-{}] "
                                 "- 30.0*dev_var_in[loc_pp] + 16.0*dev_var_in[loc_pp+{}] - dev_var_in[loc_pp+2{}])*{};\n"
            }

# deriv wise direction names
distances = {
            "calc_deriv42_x" : "dx",
            "calc_deriv42_y": "dy" ,
            "calc_deriv42_z": "dz",
            "calc_deriv42_xx":"(dx*dx)",
            "calc_deriv42_yy" : "(dy*dy)",
            "calc_deriv42_zz" : "(dz*dz)",
            "calc_deriv42_adv_x" : "dx",
            "calc_deriv42_adv_y" : "dy",
            "calc_deriv42_adv_z" : "dz"
            }

# method parameters
methodParameters = {
            "calc_deriv42_x" : "dx",
            "calc_deriv42_y": "dy" ,
            "calc_deriv42_z": "dz",
            "calc_deriv42_xx":"dx",
            "calc_deriv42_yy" : "dy",
            "calc_deriv42_zz" : "dz",
            "calc_deriv42_adv_x" : "dx",
            "calc_deriv42_adv_y" : "dy",
            "calc_deriv42_adv_z" : "dz"
            }

# method names
pd = ["calc_deriv42_x", "calc_deriv42_y", "calc_deriv42_z", "calc_deriv42_xx", "calc_deriv42_yy", "calc_deriv42_zz"]

# const variables setting
constVariables = {"calc_deriv42_x" : "idx",
                    "calc_deriv42_y": "idy",
                    "calc_deriv42_z": "idz",
                    "calc_deriv42_xx": "idx_sqrd",
                    "calc_deriv42_yy": "idy_sqrd",
                    "calc_deriv42_zz" : "idz_sqrd",
                    "calc_deriv42_adv_x" : "idx",
                    "calc_deriv42_adv_y" : "idy",
                    "calc_deriv42_adv_z" : "idz"
                    }

# output line setting for adv
advOutputLineFirstPass = "\t\toutput[pp] = ( -  3.0 * dev_var_in[loc_pp - {}]- 10.0 * dev_var_in[loc_pp] + " \
                         "18.0 * dev_var_in[loc_pp + {}] -  6.0 * dev_var_in[loc_pp + {}] + dev_var_in[loc_pp + {}]) * {};\n"

# output line setting for adv in else
advOutputLineSecondPass = "\t\toutput[pp] = ( -  dev_var_in[loc_pp - {}] +  6.0 * dev_var_in[loc_pp - {}] " \
                          "- 18.0 * dev_var_in[loc_pp - {}] + 10.0 * dev_var_in[loc_pp] " \
                          "+  3.0 * dev_var_in[loc_pp + {}]) * {};\n"

# adv if output
advFirstPassParam = {
            "calc_deriv42_adv_x" : [1, 1, 2, 3],
            "calc_deriv42_adv_y" : [16, 16, "2*16", "3*16"],
            "calc_deriv42_adv_z" : ["16*16", "16*16", "2*16*16", "3*16*16"]
            }

# adv else output
advSecondPassParam = {
            "calc_deriv42_adv_x" : [3, 2, 1, 1],
            "calc_deriv42_adv_y" : ["3*16", "2*16", "16", "16"],
            "calc_deriv42_adv_z" : ["3*16*16", "2*16*16", "16*16", "16*16"]
            }
# adv methods
adv = ["calc_deriv42_adv_x", "calc_deriv42_adv_y", "calc_deriv42_adv_z"]

# adv beta variables
advBeta = {
    "calc_deriv42_adv_x": "betax",
    "calc_deriv42_adv_y" :"betay",
    "calc_deriv42_adv_z": "betaz"
    }

tile_sz = 10

def writeToFile(declare_deriv_file, var):

    declare_deriv_file.write("{\n")
    declare_deriv_file.write("\tint tile_x = blockIdx.x%tile_size;\n")
    declare_deriv_file.write("\tint tile_y = blockIdx.x/tile_size%tile_size;\n")
    declare_deriv_file.write("\tint tile_z = blockIdx.x/tile_size/tile_size;\n")
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\tint x_offset = tile_x*{};\n".format(tile_sz))
    declare_deriv_file.write("\tint y_offset = tile_y*{};\n".format(tile_sz))
    declare_deriv_file.write("\tint z_offset = tile_z*{};\n".format(tile_sz))
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\tint i_thread = threadIdx.x % {};\n".format(tile_sz))
    declare_deriv_file.write("\tint j_thread = threadIdx.x/{}%{};\n".format(tile_sz, tile_sz))
    declare_deriv_file.write("\tint k_thread = threadIdx.x/{}/{};\n".format(tile_sz, tile_sz))
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\tint i = i_thread + x_offset;\n")
    declare_deriv_file.write("\tint j = j_thread + y_offset;\n")
    declare_deriv_file.write("\tint k = k_thread + z_offset;\n")
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\t// associated shared memory location\n")
    declare_deriv_file.write("\tint i_shared = i_thread + 3;\n")
    declare_deriv_file.write("\tint j_shared = j_thread + 3;\n")
    declare_deriv_file.write("\tint k_shared = k_thread + 3;\n")
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\tif (i>=host_sz_x-{}) {{ return; }}\n".format(dirnVisePositions[var][0]))
    declare_deriv_file.write("\tif (i<{}) return; \n".format(dirnVisePositions[var][0]))
    declare_deriv_file.write("\tif (j>=host_sz_y-{}) {{ return; }}\n".format(dirnVisePositions[var][1]))
    declare_deriv_file.write("\tif (j<{}) return; \n".format(dirnVisePositions[var][1]))
    declare_deriv_file.write("\tif (k>=host_sz_z-{}) {{ return; }}\n".format(dirnVisePositions[var][2]))
    declare_deriv_file.write("\tif (k<{}) return; \n".format(dirnVisePositions[var][2]))
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\tint nx = host_sz_x; \n")
    declare_deriv_file.write("\tint ny = host_sz_y;\n")
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\tconst double {} = 1.0/{};\n".format(constVariables[var], distances[var]))
    declare_deriv_file.write("\tconst double {} = {} / 12.0;\n".format(idx_by[var], constVariables[var]))
    declare_deriv_file.write("\n")

    declare_deriv_file.write("\tint pp = IDX(i, j, k);\n")
    declare_deriv_file.write(
        "\tint loc_pp = k_shared*{}*{} + j_shared*{} + i_shared;\n".format(tile_sz + 6, tile_sz + 6, tile_sz + 6))
    declare_deriv_file.write("\n")

with open("generated/deviceDerivsShared.cu", "w") as declare_deriv_file:
    addHeader(declare_deriv_file, "bssn/cuda_gr/cuda_src")

    declare_deriv_file.write("# include \"deviceDerivs.cuh\"\n")
    declare_deriv_file.write("\n")
    methodSignature = "__device__ void {}(int tile_size, double * output, double * dev_var_in, " \
                      "const int u_offset, double {}, const unsigned int host_sz_x, const unsigned int host_sz_y, " \
                      "const unsigned int host_sz_z, int bflag)\n"
    # this should be taken from the user as a parameter
    for var in pd:
        declare_deriv_file.write(methodSignature.format(var, methodParameters[var]))
        writeToFile(declare_deriv_file, var)

        declare_deriv_file.write(outputLine[var].format(
                                stencils[var][0], stencils[var][1], stencils[var][1], stencils[var][0], stencils[var][2]))

        declare_deriv_file.write("\n")

        declare_deriv_file.write("}\n")

        declare_deriv_file.write("\n")

    advMethodSignature = "__device__ void {}(int tile_size, double * output, double * dev_global_var_in," \
                         " double * dev_var_in, int u_offset, double {}, int {}, const unsigned int host_sz_x, " \
                         "const unsigned int host_sz_y, const unsigned int host_sz_z, int bflag)\n"

    for var in adv:
        declare_deriv_file.write(advMethodSignature.format(var, methodParameters[var], advBeta[var]))
        writeToFile(declare_deriv_file, var)

        declare_deriv_file.write("\tif (dev_global_var_in[{} + pp] > 0.0) {{\n".format(advBeta[var]))


        declare_deriv_file.write(advOutputLineFirstPass.format(
            advFirstPassParam[var][0], advFirstPassParam[var][1], advFirstPassParam[var][2], advFirstPassParam[var][3],
            idx_by[var]))
        declare_deriv_file.write("\t}\n")

        declare_deriv_file.write("\telse {\n")
        declare_deriv_file.write(advOutputLineSecondPass.format(
            advSecondPassParam[var][0], advSecondPassParam[var][1], advSecondPassParam[var][2], advSecondPassParam[var][3],
            idx_by[var]))

        declare_deriv_file.write("\t}\n")

        declare_deriv_file.write("\n")
        declare_deriv_file.write("}\n")

        declare_deriv_file.write("\n")