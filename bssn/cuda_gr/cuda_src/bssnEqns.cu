/**
 * Created on: Sep 21, 2018
 * 		Author: Akila, Eranga, Eminda, Ruwan
 **/
 
#include "bssnEqns.cuh"
#define thread_load_rhs 1

__constant__ double CUDA_ETA_CONST=0.1;
__constant__ double CUDA_ETA_R0=0.1;
__constant__ double CUDA_ETA_DAMPING_EXP=0.1;
__constant__ unsigned int lambda[4]={1,2,3,4};
__constant__ double lambda_f[2]={0.8,0.9};

#include "rhs_unstaged_shared.cuh"


__global__ void cuda_bssn_eqns_points_b_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        
        b_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );
        
    }
    
}

__global__ void cuda_bssn_eqns_points_chi_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        chi_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );
        
    }
    
}
__global__ void cuda_bssn_eqns_points_gt_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        gt_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );
    }
    
}

__global__ void cuda_bssn_eqns_points_At_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*32 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        At_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

       
    }
    
}

__global__ void cuda_bssn_eqns_points_K_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*64 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        K_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_CalGt_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        CalGt(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs1(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s1_(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );
    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs2(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s2_(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs3(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s3_(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs4(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s4_(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs6(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s6_(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs5(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s5_(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );
      

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs7(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s7_(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*256 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        Gt_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );
        

    }
    
}
__global__ void cuda_bssn_eqns_points_a_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*256 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        a_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );
        

    }
    
}
__global__ void cuda_bssn_eqns_points_B_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"
    )
{
    int thread_id = blockIdx.x*128 + threadIdx.x;

    for (int id = thread_id*thread_load_rhs; id<(thread_id+1)*thread_load_rhs; id++){
        int i = id%(host_sz_x-6) + 3;
        int j = ((id/(host_sz_x-6))%(host_sz_y-6)) + 3;
        int k = (id/(host_sz_z-6)/(host_sz_y-6)) + 3;

        if (k>=host_sz_z-3) return;

        double z = pmin_z + hz*k;
        double y = pmin_y + hy*j;
        double x = pmin_x + hx*i;

        int pp = i + (host_sz_x)*(j + (host_sz_y)*k);
        double r_coord = sqrt(x*x + y*y + z*z);
        double eta = CUDA_ETA_CONST;
        if (r_coord >= CUDA_ETA_R0) {
            eta *= pow( (CUDA_ETA_R0/r_coord), CUDA_ETA_DAMPING_EXP);
        }
        
        B_rhs(pp,eta, dev_var_in, dev_var_out,
            #include "args_derivs_offsets.h"
            ,
            #include "args_staged.h"
        );

    }
    
}

void calc_bssn_eqns_kernel_wrapper(double * dev_var_in, double * dev_var_out, const unsigned int * sz, const double * pmin, double hz, double hy, double hx, cudaStream_t stream,
    #include "para_derivs_offsets.h"
    ,
    #include "para_staged.h"

)
{
    double pmin_x = pmin[0];
    double pmin_y = pmin[1];
    double pmin_z = pmin[2];

    const unsigned int host_sz_x = sz[0];
    const unsigned int host_sz_y = sz[1];
    const unsigned int host_sz_z = sz[2];

    int total_points = ceil(1.0*(sz[2]-6)*(sz[1]-6)*(sz[0]-6)/thread_load_rhs);
    
    int threads_per_block_At_rhs=32;
    int threads_per_block_K_rhs=64;
    int number_of_blocks_128_threads = ceil(1.0*total_points/128);
    int number_of_blocks_256_threads = ceil(1.0*total_points/256);
    int number_of_blocks_512_threads = ceil(1.0*total_points/512);
    int number_of_blocks_K_rhs = ceil(1.0*total_points/threads_per_block_K_rhs);
    int number_of_blocks_At_rhs = ceil(1.0*total_points/threads_per_block_At_rhs);

    cuda_bssn_eqns_points_b_rhs<<< number_of_blocks_128_threads, 128, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_first_part Kernel launch failed");

    cuda_bssn_eqns_points_chi_rhs<<< number_of_blocks_512_threads, 512, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_first_part Kernel launch failed");

    cuda_bssn_eqns_points_gt_rhs<<< number_of_blocks_128_threads, 128, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_At_rhs<<< number_of_blocks_At_rhs, threads_per_block_At_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_CalGt_rhs Kernel launch failed");
    
    cuda_bssn_eqns_points_K_rhs<<< number_of_blocks_K_rhs, threads_per_block_K_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_CalGt_rhs<<< number_of_blocks_128_threads, 128, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs1<<< number_of_blocks_256_threads, 256, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_At_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs2<<< number_of_blocks_256_threads, 256, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_At_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs3<<< number_of_blocks_512_threads, 512, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_At_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs4<<< number_of_blocks_128_threads, 128, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs5<<< number_of_blocks_256_threads, 256, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs6<<< number_of_blocks_128_threads, 128, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs7<<< number_of_blocks_256_threads, 256, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");


    cuda_bssn_eqns_points_Gt_rhs<<< number_of_blocks_256_threads, 256, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");
    
    cuda_bssn_eqns_points_a_rhs<<< number_of_blocks_256_threads, 256, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_B_rhs<<< number_of_blocks_256_threads, 256, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "args_derivs_offsets.h"
        ,
        #include "args_staged.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    
}