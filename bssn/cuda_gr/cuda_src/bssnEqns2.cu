#include "bssneqn_solve.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

__constant__ double ETA_CONST=0.1;
__constant__ double ETA_R0=0.1;
__constant__ double ETA_DAMPING_EXP=0.1;
__constant__ unsigned int lambda[4]={1,2,3,4};
__constant__ double lambda_f[2]={0.8,0.9};

__device__ void a_rhs(int pp, double eta, double * dev_var_in,
    double * dev_var_out,
    #include "list_of_offset_para.h"
    ,
   #include "list_of_para.h"
   ,
    #include "list_of_staged_para.h"
){
   #include "staged_code/a_rhs.h"
}

__device__ void At_rhs(int pp, double eta, double * dev_var_in,
    double * dev_var_out,
    #include "list_of_offset_para.h"
    ,
   #include "list_of_para.h"
   ,
    #include "list_of_staged_para.h"
){
   #include "staged_code/At_rhs.h"
}

__device__ void B_rhs(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
  #include "list_of_staged_para.h"
){
   #include "staged_code/B_rhs.h"
}

__device__ void b_rhs(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/b_rhs.h"
}

__device__ void CalGt(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/CalGt.h"
}

__device__ void chi_rhs(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/chi_rhs.h"
}

__device__ void Gt_rhs_s1(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs_s1_.h"
}

__device__ void Gt_rhs_s2(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs_s2_.h"
}

__device__ void Gt_rhs_s3(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs_s3_.h"
}

__device__ void Gt_rhs_s4(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs_s4_.h"
}

__device__ void Gt_rhs_s5(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs_s5_.h"
}

__device__ void Gt_rhs_s6(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
   ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs_s6_.h"
}

__device__ void Gt_rhs_s7(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
    ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs_s7_.h"
}

__device__ void Gt_rhs(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
    ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/Gt_rhs.h"
}

__device__ void gt_rhs(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
    ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/gt_rhs.h"
}

__device__ void K_rhs(int pp, double eta, double * dev_var_in,
   double * dev_var_out,
   #include "list_of_offset_para.h"
    ,
  #include "list_of_para.h"
  ,
   #include "list_of_staged_para.h"
){
   #include "staged_code/K_rhs.h"
}
__global__ void cuda_bssn_eqns_points_first_part(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "list_of_offset_para.h"
    ,
    #include "list_of_para.h"
    ,
    #include "list_of_staged_para.h"
    )
{
    int thread_id = blockIdx.x*threads_per_block_rhs + threadIdx.x;

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
        double eta = ETA_CONST;
        if (r_coord >= ETA_R0) {
            eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);
        }
        
        a_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        b_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        chi_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        
    }
    
}

__global__ void cuda_bssn_eqns_points_gt_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "list_of_offset_para.h"
    ,
    #include "list_of_para.h"
    ,
    #include "list_of_staged_para.h"
    )
{
    int thread_id = blockIdx.x*threads_per_block_rhs + threadIdx.x;

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
        double eta = ETA_CONST;
        if (r_coord >= ETA_R0) {
            eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);
        }
        gt_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
    }
    
}

__global__ void cuda_bssn_eqns_points_At_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "list_of_offset_para.h"
    ,
    #include "list_of_para.h"
    ,
    #include "list_of_staged_para.h"
    )
{
    int thread_id = blockIdx.x*threads_per_block_At_rhs + threadIdx.x;

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
        double eta = ETA_CONST;
        if (r_coord >= ETA_R0) {
            eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);
        }
        
        At_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );

        K_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );

        Gt_rhs_s5(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        Gt_rhs_s6(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        Gt_rhs_s7(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
    }
    
}

__global__ void cuda_bssn_eqns_points_CalGt_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "list_of_offset_para.h"
    ,
    #include "list_of_para.h"
    ,
    #include "list_of_staged_para.h"
    )
{
    int thread_id = blockIdx.x*threads_per_block_rhs + threadIdx.x;

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
        double eta = ETA_CONST;
        if (r_coord >= ETA_R0) {
            eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);
        }
        CalGt(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs_second_part(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "list_of_offset_para.h"
    ,
    #include "list_of_para.h"
    ,
    #include "list_of_staged_para.h"
    )
{
    int thread_id = blockIdx.x*threads_per_block_rhs + threadIdx.x;

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
        double eta = ETA_CONST;
        if (r_coord >= ETA_R0) {
            eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);
        }
        Gt_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        B_rhs(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        

    }
    
}

__global__ void cuda_bssn_eqns_points_Gt_rhs(double * dev_var_in, double * dev_var_out, 
    const unsigned int host_sz_x, const unsigned int host_sz_y, const unsigned int host_sz_z,  
    double pmin_x, double pmin_y, double pmin_z, 
    double hz, double hy, double hx, 
    #include "list_of_offset_para.h"
    ,
    #include "list_of_para.h"
    ,
    #include "list_of_staged_para.h"
    )
{
    int thread_id = blockIdx.x*threads_per_block_rhs + threadIdx.x;

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
        double eta = ETA_CONST;
        if (r_coord >= ETA_R0) {
            eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);
        }
        
        Gt_rhs_s1(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        Gt_rhs_s2(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        Gt_rhs_s3(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );
        Gt_rhs_s4(pp, eta, dev_var_in, dev_var_out,
            #include "list_of_offset_args.h"
            ,
            #include "list_of_args.h"
            ,
            #include "list_of_staged_args.h"
        );

    }
    
}

void calc_bssn_eqns(double * dev_var_in, double * dev_var_out, const unsigned int * sz, const double * pmin, double hz, double hy, double hx, cudaStream_t stream,
#include "list_of_offset_para.h"
, 
#include "list_of_para.h"
,
#include "list_of_staged_para.h"

)
{
    double pmin_x = pmin[0];
    double pmin_y = pmin[1];
    double pmin_z = pmin[2];

    const unsigned int host_sz_x = sz[0];
    const unsigned int host_sz_y = sz[1];
    const unsigned int host_sz_z = sz[2];

    int total_points = ceil(1.0*(sz[2]-6)*(sz[1]-6)*(sz[0]-6)/thread_load_rhs);

    int number_of_blocks = ceil(1.0*total_points/threads_per_block_rhs);
    int number_of_blocks_At_rhs = ceil(1.0*total_points/threads_per_block_At_rhs);

    cuda_bssn_eqns_points_first_part<<< number_of_blocks, threads_per_block_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "list_of_offset_args.h"
        ,
        #include "list_of_args.h"
        ,
        #include "list_of_staged_args.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_first_part Kernel launch failed");

    cuda_bssn_eqns_points_gt_rhs<<< number_of_blocks, threads_per_block_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "list_of_offset_args.h"
        ,
        #include "list_of_args.h"
        ,
        #include "list_of_staged_args.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_CalGt_rhs<<< number_of_blocks, threads_per_block_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "list_of_offset_args.h"
        ,
        #include "list_of_args.h"
        ,
        #include "list_of_staged_args.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_CalGt_rhs Kernel launch failed");
    
    cuda_bssn_eqns_points_Gt_rhs<<< number_of_blocks, threads_per_block_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "list_of_offset_args.h"
        ,
        #include "list_of_args.h"
        ,
        #include "list_of_staged_args.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");

    cuda_bssn_eqns_points_At_rhs<<< number_of_blocks_At_rhs, threads_per_block_At_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "list_of_offset_args.h"
        ,
        #include "list_of_args.h"
        ,
        #include "list_of_staged_args.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_At_rhs Kernel launch failed");

    cuda_bssn_eqns_points_Gt_rhs_second_part<<< number_of_blocks, threads_per_block_rhs, 0, stream >>>(dev_var_in, dev_var_out, 
        host_sz_x, host_sz_y, host_sz_z, 
        pmin_x, pmin_y, pmin_z, 
        hz, hy, hx, 
        #include "list_of_offset_args.h"
        ,
        #include "list_of_args.h"
        ,
        #include "list_of_staged_args.h"
    ); 
    CHECK_ERROR(cudaGetLastError(), "cuda_bssn_eqns_points_Gt_rhs Kernel launch failed");
}
