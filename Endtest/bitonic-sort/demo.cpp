#include "bitonic_sort.h"
#include <cstdio>
#include <cstdlib> 
#include <cassert>
#include <climits> 
#include <sys/mman.h>

#include <ctime> 
#include <cstring>
#include <algorithm>
#include <iostream>


//OpenCL
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <CL/cl.h>
#include <time.h>




using namespace std;

/////////////////////

#include <time.h>
#include <assert.h>
#include <iostream>

#define ASCENDING true
#define DESCENDING false

void RecursiveSort(const int lo, const int n, int * a, const bool dir);
void RecursiveMerge(const int lo, const int n, int * a, const bool dir);

/* a - array to be sorted 
length - length of the array

This function takes on the array and legnth of 
the array and setups the recursive sort procedure */

void Bitonicsort_CPU(int * a, const int length)
{
  RecursiveSort(0, length, a, ASCENDING);
}

/*
lo - start index of subsection of the array
n - length of subsection of the array
a - array to be sorted
dir - direction of sort (ascending - descending)

RecursiveSort and RecursiveMerge
Recursively moves to the lower stage of the arrya 
sorts each half of the array to be a monotonic 
sequence and then merges it recursively into one

*/

void RecursiveSort(const int lo, const int n, int * a, const bool dir)
{
    if (n>1)
    {
        int m=n/2;
        RecursiveSort(lo, m,   a, ASCENDING);
        RecursiveSort(lo+m, m, a, DESCENDING);
        RecursiveMerge(lo, n,  a, dir);
    }
}

void RecursiveMerge(const int lo, const int n, int * a, const bool dir)
{
    if (n>1)
    {
        int m=n/2;
        for (int i=lo; i<lo+m; i++)
    {
      if (dir  == ( a[i] > a[i+m] )) 
      {
        float  t=a[i];
        a[i]=a[i+m];
        a[i+m]=t;
      }
    }
        RecursiveMerge(lo, m, a, dir);
        RecursiveMerge(lo+m, m, a, dir);
    }
}



/////////////////////
void print(int *tab, int n){
  for (int i = 0; i < n; ++i) {
    printf("%d \n", tab[i]);
  }
}

unsigned int logarithm2( unsigned int x )
{
  unsigned int ans = 0 ;
  while( x>>=1 ) ans++;
  return ans ;
}



void uOpenCL(int* to_sort, int size){
    // Retrieving Platforms
  cl_int status;
  cl_uint numPlatforms;
  cl_platform_id *platforms = NULL;

  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  if (status != 0)
  {
    printf("Error: %s\n", status);
  }

  // Retrieving Devices
  cl_uint numDevices;
  cl_device_id *devices = NULL;

  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
  if (status != 0)
  {
    printf("Error: %s\n", status);
  }


  // Creating Context
  cl_context context = NULL;
  context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
  if (status != 0)
  {
    printf("Error: %s\n", status);
  }

  // Creating a Command Queue
  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
  if (status != 0)
  {
    printf("Error: %s\n", status);
  }


  // Setup device memory
  cl_mem d_A;
  d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)* size, to_sort, &status);
  if (status != 0)
  {
    printf("Error: %s\n", status);
  }


  //Load and build OpenCL kernel
  cl_kernel clKernel;
  const char fileName[] = "./bitonic_kernel_random.cl";
  size_t sourceSize;
  char *source_str;
  FILE* fp = fopen(fileName, "r");
  if (!fp)
  {
    printf("Error while loading the source code %s\n", fp);
    exit(1);
  }
  source_str = (char *)malloc(sizeof(char) * 1000000);
  sourceSize = fread(source_str, 1, 1000000, fp);
  fclose(fp);



  cl_program clProgram;
  clProgram = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t*)&sourceSize, &status);
  if (status != 0)
  {
    printf("Error: %d\n", status);
  }

  status = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
  if (status != 0)
  {
    printf("Error: %d\n", status);
  }

  clKernel = clCreateKernel(clProgram, "bitonic", &status);
  if (status != 0)
  {
    printf("Error: %d\n", status);
  }

  cl_int n = size;
  cl_int l = logarithm2(size);
  status = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_A);
  status = clSetKernelArg(clKernel, 1, sizeof(int), (void *)&n);
  status = clSetKernelArg(clKernel, 2, sizeof(int), (void *)&l);
  if (status != 0)
  {
    printf("Error: %d\n", status);
  }

  size_t  globalWorkSize[2];
  size_t localWorkSize[2];
  int sizen = min(size,2048);
  localWorkSize[0] = sizen/2;
  localWorkSize[1] = 1;
  globalWorkSize[0] = sizen/2;
  globalWorkSize[1] = 1;
  // printf("123\n");


  cl_event profiling_event;

  status = clEnqueueNDRangeKernel(cmdQueue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &profiling_event);
  if (status != 0)
  {
    cout<<status<<" "<<size<<" "<<localWorkSize[0];
    printf("Error:status \n");
  }

  clWaitForEvents(1, &profiling_event);

  // cl_ulong start_time;
  // cl_ulong finish_time;
  // clGetEventProfilingInfo(profiling_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
  // clGetEventProfilingInfo(profiling_event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);


  // cl_ulong total_time = finish_time - start_time;
  // printf("Start time in nanoseconds = %lu\n", start_time);
  // printf("End time in nanoseconds = %lu\n", finish_time);
  // printf("Total time in nanoseconds = %lu\n", total_time);
  

  //Retrieve result from device
  status = clEnqueueReadBuffer(cmdQueue, d_A, CL_TRUE, 0, sizeof(int)* size, to_sort, 0, NULL, NULL);
  if (status != 0)
  {
    printf("Error: %s\n", status);
  }

  // //Print out the results
  // printf("Printing only first 10 results\n");
  // for (int i = 0; i < 10; i++)
  // {
  //   printf("%d\n", to_sort[i]);
  // }
  // free(to_sort);

  clReleaseMemObject(d_A);

  free(devices);
  free(platforms);

  clReleaseKernel(clKernel);
  clReleaseProgram(clProgram);
  clReleaseCommandQueue(cmdQueue);


}

void comparesorts(int n) {
  // int n = 1024*1024*50;
  cout << "Size "<< n<< ":"<<endl;

  int *c1 = (int*) malloc(n*sizeof(int));
  int *c2 = (int*) malloc(n*sizeof(int));
  std::clock_t start;
  int* d;

  //rand
  for (int j=0; j<n; ++j) {
      c1[j] = rand();
  }
  memcpy(c2, c1, n*sizeof(int));
  start = std::clock();
  d = bitonic_sort(c1, n);
  std::cout << "Time for bitonic_CUDA: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  start = std::clock();
  uOpenCL(c1, n);
  std::cout << "Time for bitonic_OpenCL: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  start = std::clock();
  Bitonicsort_CPU(c1, n);
  std::cout << "Time for Bitonicsort_CPU: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  start = std::clock();
  sort(c2, c2 + n);
  std::cout << "Time for STL: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl << endl;
  free(d);
  free(c1);
  free(c2);
}

int main(){
  srand( time( NULL ) );
  cout << "===================" <<endl; 
  cout << "multiples of 1024:" << endl;
  cout << "===================" <<endl; 
  cout << endl;

  for (int i = 1; i <= 1024*64; i *= 2) {
    comparesorts(i*1024);
  }

  cout << endl;
  cout << "===================" <<endl; 
  cout << "not multiples of 1024:" << endl;
  cout << "===================" <<endl; 
  cout << endl;

  for (int i = 1; i < 1024*64; i *= 2) {
    comparesorts(i*1025);
  }

  cout << endl;
  cout << "===================" <<endl; 
  cout << "big numbers:" << endl;
  cout << "===================" <<endl; 
  cout << endl;

  for (int i = 1024*64; i <= 1024*256; i *= 2) {
    comparesorts(i*1024);
  }
  
  return 0;
}