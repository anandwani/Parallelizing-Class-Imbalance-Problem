import math
import time
import numpy as np
import pycuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from scipy import signal
import sys 
import pycuda.autoinit
import matplotlib
import matplotlib.pyplot as plt 

class VectorDistance:

   def __init__(self):
      """ 
      Attributes for instance of EncoderDecoder module
      """
      self.mod = self.getSourceModule()
      pass
   
   def getSourceModule(self):
      # kernel code wrapper
      kernelwrapper = """ 
      #include<stdio.h>

      /* Vector Distance GPU Naive Method  ---> 

         matrixSamples - inputMatrix i.e. rowsCount * featureCount 
         result - to store the output of vectorDistance

         ** if 5 rows are there in the input table, we need 25 rows of output.
            i.e. Dij is the distance between ith point to jth point. 
            So, for every point, we need it distance between all the other points. 
            Hence threads are invoked in both X & Y direction. 
      */

      __global__ void vectorDistance_gpuNaive(float *matrixSamples, 
                                              float *result, 
                                              int rowsCount, 
                                              int featuresCount)
      {
         /*
            ** threads in X and Y direction
         */
         int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         int idy = blockIdx.y * blockDim.y + threadIdx.y; 

         /* 

            ** If ith thread is same as jth thread in 2-d matrix, then the calculation
               is like D11 i.e. distance of 1st point with the 1st point only. Hence
               it can be directly written as zero

            ** else
                  we will be calculating the eucidean distance between the points 
                  and storing in the result matrix. 
         */

         if(idx<rowsCount && idy < rowsCount && idx!=idy) 
         {
            float euclideanDistance = 0.0;
            for(int k=0;k<featuresCount;k++)
            {
               euclideanDistance += pow((matrixSamples[idx*featuresCount+k]-
                                         matrixSamples[idy*featuresCount+k]),2);
            }
            result[idx*rowsCount+idy] = sqrt(euclideanDistance);
         }
         if(idx<rowsCount && idy < rowsCount && idx==idy)
         {  
            result[idx*rowsCount+idy] = 0.0;
         } 
      }
      /* <---- Vector Distance GPU Naive Method */

      """
      # you can either use a string or save the kernel in kernel.cu file and reference it here.
      # Compile the kernel code when an instance
      # of this class is made. 
      return SourceModule(kernelwrapper)

   '''
      VectorDitance GPU Naive 
         - method to invoke the kernel. 
         _matrix - dataset sample with rowsCount * featuresCount

         1) starting time event
         2) copying matrix of input and result to to gpu
         3) Invokation of threads based on the logic:
            --> if n*n <=1024, then we just need 1 block of threads for the final output
            --> else we will need blocks = math.ceil(rowsCount/32) in both X & Y direction. 
         4) Invoking the kernel : vectorDistance_gpuNaive and getting back the results.
         5) end the time event

   '''
   def vectorDistance_gpu(self,_matrix, rowsCount,featuresCount):
      
      startWithMemTransfer=cuda.Event()
      endWithMemTransfer=cuda.Event()
      startWithMemTransfer.record()

      gpu_matrix = gpuarray.to_gpu(_matrix)

      _resultMatrix= np.random.rand(rowsCount,rowsCount).astype(np.float32)
      gpu_result  = gpuarray.to_gpu(_resultMatrix)
      #_result = gpuarray.zeros_like(gpu_result)

      vdResult = np.empty_like(_matrix)
      rowsCountValue = np.uintc(rowsCount)
      featuresCountValue = np.uintc(featuresCount)

      if(rowsCount * rowsCount <= 1024):
         blockDim_X = rowsCount
         blockDim_Y = rowsCount
         gridDim_X  = 1
         gridDim_Y  = 1
      else:
         blockDim_X = 32
         blockDim_Y = 32
         gridDim_X = math.ceil(rowsCount/32)
         gridDim_Y = math.ceil(rowsCount/32)

      func = self.mod.get_function("vectorDistance_gpuNaive")

      func(gpu_matrix,
            gpu_result,
            rowsCountValue,
            featuresCountValue,
            block = (blockDim_X,blockDim_Y,1),
            grid = (gridDim_X,gridDim_Y,1)
           )

      vdResult = gpu_result.get()

      endWithMemTransfer.record()
      endWithMemTransfer.synchronize()
      timeWithMemTransfer=startWithMemTransfer.time_till(endWithMemTransfer)

      return vdResult,timeWithMemTransfer

   '''
      VectorDitance Python Serial Implementation  
         matrixSamples - dataset sample with rowsCount * featuresCount
         result - to store the output. 

         1) starting time event
         2) Logic to impute vector distance
            --> for all elements in array:
               --> for all elements in array:
                  --> if i==j then distance is 0
                  --> else calculate the eucildean distance bwteen Di,j
         3) end the time event

   '''
   def vectorDistance_python(self,matrixSamples,rowsCount,featuresCount,result):
      start = time.time()
      for i in range(0,rowsCount):
         for j in range(0, rowsCount):
            if(i==j):
               result[i][j] = 0
            else:
               euclideanDistance=0;
               for k in range(0,featuresCount):
                  euclideanDistance += math.pow((matrixSamples[j][k]-matrixSamples[i][k]),2)
               result[i][j]=math.sqrt(euclideanDistance)
      end = time.time()
      # converting to milliseconds.
      time_ = (end-start)*1000
      return result,time_   

'''
   Main function to invoke both vector distance 
   serially on python and parallely on GPU, for 
   various sizes # 64,256,1024,2048,4096,8192.

'''
if __name__ == "__main__":

   vd = VectorDistance()

   size = [64,256,1024,2048,4096,8192]
   timeVDSerial = []
   timeVDGPU = []

   for i in range(len(size)):
      numberofRows = size[i]
      features =8

      #-----------------Input Matrix ---------------------------------------------------->
      '''
      matrixSamples = np.array([[1,3,4,5,4],
                                   [4,3,2,5,7],
                                   [5,9,2,8,6],
                                   [1,2,4,6,7],
                                   [6,5,6,7,2]
                                  ])

      '''
      matrixSamples = np.random.rand(numberofRows,features)
      matrixSamples = matrixSamples.astype(np.float32)

      print("Number of Rows - ",numberofRows)
      print("Features - ",features)

      #print("Input Matrix\n")
      #print(matrixSamples)

      print("Shape of Input Matrix=",matrixSamples.shape)
      
      #<-------------------------------------------------------------------------------------

      #------------ Python Naive Vector Distance -------------------------------------------->
      resultVD_Python = np.random.rand(matrixSamples.shape[0],
                                       matrixSamples.shape[0])
      resultVD_Python = resultVD_Python.astype(np.float32)    
      
      vd_python, time_vd_python = vd.vectorDistance_python(matrixSamples,
                                                              matrixSamples.shape[0],
                                                              matrixSamples.shape[1],
                                                              resultVD_Python)

      #print("\nPython Naive Output Vector Distance\n")
      #print(vd_python)

      print("Time Taken by Python Naive Vector Distance in Seconds")
      print(time_vd_python)
      
      timeVDSerial.append(time_vd_python)
      #<------------------------------------------------------------------------------------
       
      #----------------- GPU Naive Vector Distance -----------------------------------------> 
      vd_gpu,time_vd_gpu = vd.vectorDistance_gpu(matrixSamples,
                                                    matrixSamples.shape[0],
                                                    matrixSamples.shape[1])

      #print("\nGPU Naive Output Vector Distance\n")
      #print(vd_gpu)

      print("Time Taken by GPU Naive Vector Distance in Seconds")
      print(time_vd_gpu)
    
      print("\n")
      timeVDGPU.append(time_vd_gpu)
      #<--------------------------------------------------------------------------------------

      #----------------- Comparision Vector Distan--------------------->
      if(np.allclose(vd_gpu, vd_python, atol=1e-05)):
         
         print("----------------------------------------")
         print("-------Vector Distance Matched---------")
         print("----------------------------------------")
      else:
         print("Vector Distance NOT MATCHED!")

      #------------------------------------------------------------

      print("\n\n")

   plt.xlabel("Sizes")
   plt.ylabel("Time taken (ms)")
   plt.yscale('log')
   plt.title("Time taken for generating new matrix for different sizes")
   plt.plot(size,timeVDSerial,label='Vector Distance Python',color="red",marker='o')
   plt.plot(size,timeVDGPU,label='Vector Distance GPU',color="orange",marker='o')
   plt.legend()

