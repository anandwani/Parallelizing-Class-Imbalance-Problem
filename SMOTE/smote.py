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

class Smote:

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
            So, for every point, we need to compute distance between all the other points. 
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
      /* Sorting method used in k-nn --->  
         ** Here is an example of what this method is doing
            distancePerIdx[] = [3.5, 5.53, 2.3 , 5.2, 9.4]
            positionsAsDistance[] = [0,1,2,3,4]
            then we need to sort the distancePerIdx by size, but at the same time
            sort the positions array also to which it belongs. i.e. the final
            result will be
            distancePerIdx[] = [2.3, 3.5, 5.2, 5.53, 9.4]
            positionsAsDistance[] = [2,0,3,1,4]
         ** Please note, we choose bubble sort over quick sort here
            because, quick sort involves recurssion
            and we were worried that the stack could overflow in 
            in case of large matrices like 8K or 16K. Hence we decided
            to go with bubble sort. 
      */
      __device__ void sortAsPerDistance(float *distancePerIdx, 
                                        int *positionsAsDistance, 
                                        int rowsCount)
      {
         int m,n;
            for (m = 0; m < rowsCount-1; m++)  
            {   
               for (n = 0; n < rowsCount-m-1; n++)
               {
                  /*
                     ** condition on the distance array but sorting both distance
                        and position array.
                  */
                  if (distancePerIdx[n] > distancePerIdx[n+1])
                  {
                     float temp = distancePerIdx[n];
                     distancePerIdx[n] = distancePerIdx[n+1];
                     distancePerIdx[n+1] = temp;
         
                     int t = positionsAsDistance[n];
                     positionsAsDistance[n] = positionsAsDistance[n+1];
                     positionsAsDistance[n+1] = t;
                  }
               }
            }
      } /* <--- Sorting method used in k-nn */
      /*
         ** Since we want to sort, we have to implcitly declare the rowSize here. 
            The rowSize should match the maximum elements in the input matrix. 
         ** Please note, the RowSize is based out of the 
            vectorDistance output matrix, which will be n*n, where n is number of 
            rows in input
      */
      #define ROWSIZE 8192 // Change here the value as defined in main for row samples in matrix.
      /* knn Gpu Naive Method -->
         ** Both vectorDistance output and input matrix are passed as input to knn_gpu_naive
         ** result is to store the output from knn i.e. matrix with new points 
         ** rowsCount & featureCount are the rows and columns for the input matrix
         ** k is the value for "k" in  k-nearest neighbor algorithm.
         ** The algorithm is parallelized in 1 direction i.e. X axis. 
            So each thread is working row-wise on the vector distance matrix
            to impute the k-nearest neighbor for that point. 
            For example, for point 1: if distanceArray=[0, 1.4, 5.2, 2.4, 8.9] 
            is an array achievd through vector distance algorithm, i.e. distance of 
            point 1 with 1,2,3,4,5 respectively, 
            then for a value of k=2, the below code first stores it in the local
            array and then sorts the array to find the k-nearest neigbor to point 1. 
            Once nearest neighbors are found, we check to what position is belongs, as 
            we are switching the value of positions too and then pick the points 
            accordingly from the input matrix. 
            As in above example, it will maintain 2 arrays
            distanceArray=[0, 1.4, 5.2, 2.4, 8.9]
            positionArray = [1,2,3,4,5]
            then after sorting
            distanceArray =[0, 1.4, 2.4, 5.2, 8.9]
            positionArray = [1,2,4,3,5]
            so the k=2 nearest points are 2 & 4 to point 1. 
            Please note, there will always be an entry of 0 at the first position which signifies
            Distance of first point with itself. We will have to ignore this value and start
            from 1st index.
      */
      __global__ void knn_gpuNaive(float *matrixSamples, 
                                   float *vectorDistance, 
                                   float *result, 
                                   int rowsCount, 
                                   int featuresCount,int k)
      {
         /*
            ** threads invoked in one direction. If n rows, n threads are invoked.
               The threads are working on vector distance output, to find
               k-nearest neighbor for each point.
         */
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         /*
            ** storing the distances and positions as explained above, to 
               be able to sort the array, to find k-minimum. 
         */
         float distancePerIdx[ROWSIZE];
         int positionsAsDistance[ROWSIZE];  
         if(idx<rowsCount)
         {
            for(int j=0;j<rowsCount;j++)
            {
               distancePerIdx[j]=vectorDistance[idx*rowsCount+j];
               positionsAsDistance[j] = j;
            }
            sortAsPerDistance(distancePerIdx,positionsAsDistance,rowsCount);
            /*
               ** Once we have the sorted row for the thread,  we just have
                  to compute the new point by taking the average of k-nearest point
                  with the point of minority class.
               ** In the above example, point 1 had 2 nearest neigbors i.e. 2 & 4. 
                  So, avg will be taken between 1 & 2 and 1 & 4 to compute
                  new points.
               ** Please note, we started with J=1 here
            */
            for(int j=1;j<k+1;j++)
            {
               for(int featureIndx=0;featureIndx<featuresCount;featureIndx++)
               {
                  if((idx+(rowsCount*(j-1)))<(rowsCount*k))
                  {
                     result[(idx+(rowsCount*(j-1)))*featuresCount+featureIndx] = 
                                 (
                                    matrixSamples[idx*featuresCount+featureIndx] + 
                                    matrixSamples[((positionsAsDistance[j]))*featuresCount+featureIndx]
                                 )/2;
                  }
               }
            }
         }
      }
      /* <-- knn Gpu Naive Method */
      /* Shared Memory GPU Implementation for K-Nearest Neighbor --> 
         **  
            ** The implementation using shared memory remains nearly the same 
               with naive, except some points will be accessed from the shared 
               memory rather than inducing an I/O everytime. 
            ** For the shared memory implementation, the idea was to see if we can achieve
               better elapsed time as the GPU NAIVE KNN was only giving 2X improverment.
            ** We brainstormed alot, and came up with an implementation, which is giving better 
               elapsed time when compared to GPU Naive. However, the benefit was not that much. 
               Below is the explanation of the  tehcnique and the reason as to why we were not 
               able to see much benefit.
            ** Technique
                  - For imputing new features, let's take 2 points 1 & 3. Now with k=2, lets
                  assume our nearest neighbor were (5,6) for 1 and (5,8) for point 3. Now for computing
                  new points through 1 & 3, we need to access 5th Point twice from the input
                  matrix.
                  - To reduce this multiple access from the input matrix, we decided we will
                  store as many points as possible in the shared memory, so that the time taken to 
                  retrieve point 5 for both 1 & 3 is reduced.
                  - However, since there is an upper limit to the number of points that can be stored in shared
                  memory, we could only store maximum of 128 elements with 8 features each, consituting
                  1024 elements in the shared memory.
                  - So, the shared memory for every block will store first 128 elements in the shared 
                     memory which will be used in calculation of new points.
                  - However 128 on 1024 elements are not too much, and hence we need a technique to 
                    decide what can these 128 elements be. One way is to choose random points, but a better way
                    will be to see the maximum hits for each point i.e. if we can see that how many points 
                    were hit for computation of k-nearest neighbors, and how many times they were hit, 
                    then we can keep a track for 128 maximum hit points out of 1024 block elements 
                    and then store them in shared memory. We hypothesize that this technique will
                    significantly boost performance.
                  - As part of the current implementatiom, we are storing the first 128 elements from
                  each block to shared memory, and using that as part of calculation. 
                  Hence, there was some improvement in elapsed time. However, if we implement the 
                  logic to find the max hit elements per block, we are sure that we should see
                  better elapsed times. Therefore, we have drafted this as one of our future works.
      */
      #define SHARED_ROW_SIZE 128
      #define SHARED_FEATURE_SIZE 8
      __global__ void knn_gpu_shared(float *matrixSamples, 
                                     float *vectorDistance, 
                                     float *result, 
                                     int rowsCount, 
                                     int featuresCount,int k)
      {
         float distancePerIdx[ROWSIZE];
         int positionsAsDistance[ROWSIZE];      
         /*
            ** First 128 points , including max of 8 features 
               out of each block will be stored
               in the shared memory 
         */
         __shared__ float shared_mtx[SHARED_ROW_SIZE][SHARED_FEATURE_SIZE];
         int idx = threadIdx.x + blockIdx.x * blockDim.x;
         if(idx<(blockIdx.x * blockDim.x+128))
         {
            for(int j=0;j<featuresCount;j++)
            {
               shared_mtx[threadIdx.x][j] = 
                        matrixSamples[idx*featuresCount+j];
            }
         }
         __syncthreads();
         if(idx<rowsCount)
         {
            for(int j=0;j<rowsCount;j++)
            {
               distancePerIdx[j]=vectorDistance[idx*rowsCount+j];
               positionsAsDistance[j] = j;
            }
            sortAsPerDistance(distancePerIdx,positionsAsDistance,rowsCount);
            /*
               ** Now for this portion of logic, we are interested in formulating 
                  new points. So, with the point 1, if nearest neigbor were 4 & 5 for
                  k=2, then 1, 4, & 5 can be in shared memory, or any combination of the elements 
                  may or may not be in shared memory. So below is the logic to handle and 
                  retrieve the points accordingly.
                  Let us assume the points of interest are 1 & 4. Then, 
            */
            for(int j=1;j<k+1;j++)
            {
               for(int featureIndx=0;featureIndx<featuresCount;featureIndx++)
               {
                  /*
                     ** Both 1, & 4 are present in Shared Memory 
                  */
                  if(threadIdx.x<128 && 
                     (positionsAsDistance[j] > (blockIdx.x * blockDim.x)) && 
                     (positionsAsDistance[j] < (blockIdx.x * blockDim.x+ 128)) &&
                     (idx+(rowsCount*(j-1)))<(rowsCount*k)
                     )
                  {
                     result[(idx+(rowsCount*(j-1)))*featuresCount+featureIndx] = (
                           shared_mtx[threadIdx.x][featureIndx] + 
                           shared_mtx[abs((int)(blockIdx.x * blockDim.x - positionsAsDistance[j]))][featureIndx]
                           )/2; 
                  }
                  /*
                     ** Point 1 is not in shared memory but 4 is. 
                  */
                  else if(threadIdx.x>=128 && 
                          positionsAsDistance[j] > blockIdx.x * blockDim.x && 
                           positionsAsDistance[j] < blockIdx.x * blockDim.x+ 128 &&
                          (idx+(rowsCount*(j-1)))<(rowsCount*k)
                        ) 
                  {
                      result[(idx+(rowsCount*(j-1)))*featuresCount+featureIndx] = (
                           matrixSamples[idx*featuresCount+featureIndx] +    
                           shared_mtx[abs((int)(blockIdx.x * blockDim.x - positionsAsDistance[j]))][featureIndx]
                           )/2;
                  }
                  /*
                     ** Point 4 is not in shared memory but 1 is. 
                  */
                  else if(threadIdx.x<128 &&
                          positionsAsDistance[j] > blockIdx.x * blockDim.x+ 128 &&
                          (idx+(rowsCount*(j-1)))<(rowsCount*k)
                         )
                  {  
                     result[(idx+(rowsCount*(j-1)))*featuresCount+featureIndx] = (
                              shared_mtx[threadIdx.x][featureIndx] + 
                              matrixSamples[((positionsAsDistance[j]))*featuresCount+featureIndx]
                              )/2;
                  }
                  /*
                     ** Point 1 & 4 both are not in shared memory
                  */
                  else if((idx+(rowsCount*(j-1)))<(rowsCount*k))
                  {
                     result[(idx+(rowsCount*(j-1)))*featuresCount+featureIndx] = (
                              matrixSamples[idx*featuresCount+featureIndx] + 
                              matrixSamples[((positionsAsDistance[j]))*featuresCount+featureIndx]
                              )/2;
                  }
               }
            }
         }
      } /* <--- Shared Memory GPU Implementation for K-Nearest Neighbor */
  
      
      /* GPU KNN Naive with CPU SORT --->
         ** This method is where the entire matrix for vector distance is already sorted in 
            python, and now GPU is used to just compute the new points for each given
            row in the vector distance matrix according to the value of k.
      */
      __global__ void knn_gpu_cpusort(float *matrixSamples, float *vectorDistance, float *result, int rowsCount, 
                                   int featuresCount,int k, int *positionsAsDistance)
      {
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         if(idx<rowsCount)
         {
            for(int j=1;j<k+1;j++)
            {
               for(int featureIndx=0;featureIndx<featuresCount;featureIndx++)
               {
                  if((idx+(rowsCount*(j-1)))<(rowsCount*k))
                  {
                     result[(idx+(rowsCount*(j-1)))*featuresCount+featureIndx] = (
                              matrixSamples[idx*featuresCount+featureIndx] + 
                              matrixSamples[((positionsAsDistance[idx*rowsCount+j]))*featuresCount+featureIndx]
                              )/2;
                  }
               }
            }
         }
      }/* <-- GPU KNN Naive with CPU SORT */
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
         2) Logic to compute vector distance
            --> for all elements in array:
               --> for all elements in array:
                  --> if i==j then distance is 0
                  --> else calculate the eucildean distance between Di,j
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
      KNN GPU Naive 
         - method to invoke the kernel. 
         matrixSamples - dataset sample with rowsCount * featuresCount
         vectorDistance - output matrix from vector distance
         result - to store the output. 
         k - value considered for finding nearest neighbors. 
         1) starting time event
         2) copying matrix of input, vectorDistance and result to GPU
         3) Invocation of threads based on the logic:
            --> if n <=1024, then we just need 1 block of threads for the final output
            --> else we will need blocks = math.ceil(rowsCount/1024) in X direction. 
         4) Invoking the kernel : knn_gpuNaive and getting back the results.
         5) end the time event
   '''

   def knn_gpu(self,matrixSamples,vectorDistance, rowsCount,featuresCount,k,result):

      startWithMemTransfer=cuda.Event()
      endWithMemTransfer=cuda.Event()
      startWithMemTransfer.record()

      gpu_matrix = gpuarray.to_gpu(matrixSamples)
      gpu_vectorDistance = gpuarray.to_gpu(vectorDistance)

      _resultMatrix= np.random.rand(rowsCount*k,featuresCount).astype(np.float32)
      gpu_result  = gpuarray.to_gpu(_resultMatrix)

      rowsCountValue = np.uintc(rowsCount)
      featuresCountValue = np.uintc(featuresCount)
      kValue = np.uintc(k)

      if(rowsCount <= 1024):
         blockDim_X = rowsCount
         blockDim_Y = 1
         gridDim_X  = 1
         gridDim_Y  = 1
      else:
         blockDim_X = 1024
         blockDim_Y = 1
         gridDim_X = math.ceil(rowsCount/1024)
         gridDim_Y = 1

      func = self.mod.get_function("knn_gpuNaive")

      func(gpu_matrix,
            gpu_vectorDistance,
            gpu_result,
            rowsCountValue,
            featuresCountValue,
            kValue,
            block = (blockDim_X,blockDim_Y,1),
            grid = (gridDim_X,gridDim_Y,1)
           )

      knnResult = gpu_result.get()
      endWithMemTransfer.record()
      endWithMemTransfer.synchronize()
      timeWithMemTransfer=startWithMemTransfer.time_till(endWithMemTransfer)

      return knnResult,timeWithMemTransfer



   '''
      KNN GPU CPU SORT
         - method to invoke the kernel. 
         matrixSamples - dataset sample with rowsCount * featuresCount
         vectorDistance - output matrix from vector distance
         result - to store the output. 
         k - value considered for finding nearest neighbors. 
         1) starting time event
         2) Before we copy the vectorDistance output, we will first sort the 
            elements here in python, very similar to what we used in CPU 
            serial implementation of python
            #reference :-https://www.geeksforgeeks.org/python-find-the-indices-for-k-smallest-elements/
         3) Then copy the matrix of input, sorted vectorDistance matrix and result to GPU
         4) Invocation of threads based on the logic:
            --> if n <=1024, then we just need 1 block of threads for the final output
            --> else we will need blocks = math.ceil(rowsCount/1024) in X direction. 
         5) Invoking the kernel : knn_gpu_cpusort and getting back the results.
         6) end the time event
   '''
   def knn_gpu_cpusort(self,matrixSamples,vectorDistance, rowsCount,featuresCount,k,result):

      startWithMemTransfer=cuda.Event()
      endWithMemTransfer=cuda.Event()
      startWithMemTransfer.record()

      gpu_matrix = gpuarray.to_gpu(matrixSamples)
      gpu_vectorDistance = gpuarray.to_gpu(vectorDistance)

      _resultMatrix= np.random.rand(rowsCount*k,featuresCount).astype(np.float32)
      gpu_result  = gpuarray.to_gpu(_resultMatrix)

      rowsCountValue = np.uintc(rowsCount)
      featuresCountValue = np.uintc(featuresCount)
      kValue = np.uintc(k)


      distancePerRow=np.zeros(rowsCount).astype(np.float32);
      storePositionsAsDistance = np.zeros((rowsCount,rowsCount),dtype=np.int32) 
   
      
      if(k+1<rowsCount):
         requiredK = k+1 
      else:
         requiredK = rowsCount
 
      for i in range(0,rowsCount):
         for j in range(0,rowsCount):
            distancePerRow[j]=vectorDistance[i][j]
         res = sorted(range(len(distancePerRow)), key = lambda sub: distancePerRow[sub])[:requiredK]
         for m in range(0, len(res)):
            storePositionsAsDistance[i][m] = res[m]

      gpu_storedPositionsAsDistance = gpuarray.to_gpu(storePositionsAsDistance)
      
      if(rowsCount <= 1024):
         blockDim_X = rowsCount
         blockDim_Y = 1 
         gridDim_X  = 1 
         gridDim_Y  = 1 
      else:
         blockDim_X = 1024
         blockDim_Y = 1 
         gridDim_X = math.ceil(rowsCount/1024)
         gridDim_Y = 1 

      func = self.mod.get_function("knn_gpu_cpusort")

      func(gpu_matrix,
            gpu_vectorDistance,
            gpu_result,
            rowsCountValue,
            featuresCountValue,
            kValue,
            gpu_storedPositionsAsDistance,
            block = (blockDim_X,blockDim_Y,1),
            grid = (gridDim_X,gridDim_Y,1)
           )

      knnResult = gpu_result.get()

      endWithMemTransfer.record()
      endWithMemTransfer.synchronize()
      timeWithMemTransfer=startWithMemTransfer.time_till(endWithMemTransfer)

      return knnResult,timeWithMemTransfer



   '''
      KNN GPU SHARED 
         - method to invoke the kernel. 
         matrixSamples - dataset sample with rowsCount * featuresCount
         vectorDistance - output matrix from vector distance
         result - to store the output. 
         k - value considered for finding nearest neighbors. 
         1) starting time event
         2) copying matrix of input, vectorDistance and result to to gpu
         3) Invokation of threads based on the logic:
            --> if n <=1024, then we just need 1 block of threads for the final output
            --> else we will need blocks = math.ceil(rowsCount/1024) in X direction. 
         4) Invoking the kernel : knn_gpu_shared and getting back the results.
         5) end the time event
         The invocation of the knn_gpu_shared matches exactly knn_gpu_naive. The
         changes are only in the implementation of kernel.
   '''

   def knn_gpu_shared(self,matrixSamples,vectorDistance, rowsCount,featuresCount,k,result):

      startWithMemTransfer=cuda.Event()
      endWithMemTransfer=cuda.Event()
      startWithMemTransfer.record()

      gpu_matrix = gpuarray.to_gpu(matrixSamples)
      gpu_vectorDistance = gpuarray.to_gpu(vectorDistance)

      _resultMatrix= np.random.rand(rowsCount*k,featuresCount).astype(np.float32)
      gpu_result  = gpuarray.to_gpu(_resultMatrix)

      rowsCountValue = np.uintc(rowsCount)
      featuresCountValue = np.uintc(featuresCount)
      kValue = np.uintc(k)

      if(rowsCount <= 1024):
         blockDim_X = rowsCount
         blockDim_Y = 1 
         gridDim_X  = 1 
         gridDim_Y  = 1 
      else:
         blockDim_X = 1024
         blockDim_Y = 1 
         gridDim_X = math.ceil(rowsCount/1024)
         gridDim_Y = 1 

      func = self.mod.get_function("knn_gpu_shared")

      func(gpu_matrix,
            gpu_vectorDistance,
            gpu_result,
            rowsCountValue,
            featuresCountValue,
            kValue,
            block = (blockDim_X,blockDim_Y,1),
            grid = (gridDim_X,gridDim_Y,1)
           )   

      knnResult = gpu_result.get()
      endWithMemTransfer.record()
      endWithMemTransfer.synchronize()
      timeWithMemTransfer=startWithMemTransfer.time_till(endWithMemTransfer)
      
      return knnResult,timeWithMemTransfer


   '''
      K-NN Python Serial Implementation  
         matrixSamples - dataset sample with rowsCount * featuresCount
         vectorDistance - output matrix from vector distance
         result - to store the output. 
         k - value considered for finding nearest neighbors. 
         1) starting time event
         2) Logic to compute k-nearest neigbor for each row in the vector distance matrix.
            --> for all points in vector distance matrix:
               --> store all the distances computed for that point with all the other points
                   in the array 
               --> sort the first k elements of the array to get k-nearest neighbor 
                   (#Reference :-https://www.geeksforgeeks.org/python-find-the-indices-for-k-smallest-elements/)
               --> Once sorted, based on the value of k, compute new points by taking average of that point 
                   with its k-nearest neighbors. 
         3) end the time event
   '''

   def knn_python(self,matrixSamples,vectorDistance, rowsCount,featuresCount,k,result):
      start = time.time()
      newPoint=np.ones((rowsCount*k,featuresCount))
      pos=0
      for i in range(0,rowsCount):
         minValue=np.ones(rowsCount)
         minKPoints = []
         for j in range(0, rowsCount):
            minValue[j]=vectorDistance[i][j]

         

         if(k+1<rowsCount):
            requiredK = k+1
         else:
            requiredK = rowsCount

         res = sorted(range(len(minValue)), key = lambda sub: minValue[sub])[:requiredK]
         for iteration in range(1,len(res)):
            for featureIteration in range(0,featuresCount):
                newPoint[pos+(rowsCount*(iteration-1))][featureIteration] = (
                           matrixSamples[i][featureIteration] + 
                           matrixSamples[res[iteration]][featureIteration]
                           )/2
         pos=pos+1

      end = time.time()
      # converting to milliseconds.
      time_ = (end-start)*1000
      return newPoint,time_


'''
   Main function to invoke both vector distance & knn
   serially on python and parallely on GPU, for 
   various sizes # 64,256,1024,2048,4096,8192.
   ################################################
   PLEASE NOTE: you will have to update the ROWSIZE 
   variable based on the maximum value decided here.
   ROWSIZE VARIABLE IS ON LINE 135 in this file.
   ################################################
'''
if __name__ == "__main__":

   smote = Smote()

   size = [64,256,1024,2048,4096,8192]
   timeVDSerial = []
   timeVDGPU = []
   timeKnnSerial = []
   timeKnnGpuNaive = []
   timeKnnGpuCpuSort = []
   timeKnnGpuShared = []

   for i in range(len(size)):
      numberofRows = size[i]
      features =4
      k=3
      #-----------------Input Matrix ---------------------------------------------------->
      '''
      ## Sample input matrix for testing. Not used in this code.
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
      print("K - ",k)
      print("\n")

      print("Input Matrix\n")
      print(matrixSamples)

      print("\n")

      print("Shape of Input Matrix=",matrixSamples.shape)
      
      #<-------------------------------------------------------------------------------------

      #------------ Python Naive Vector Distance -------------------------------------------->
      resultVD_Python = np.random.rand(matrixSamples.shape[0],
                                       matrixSamples.shape[0])
      resultVD_Python = resultVD_Python.astype(np.float32)    
      
      vd_python, time_vd_python = smote.vectorDistance_python(matrixSamples,
                                                              matrixSamples.shape[0],
                                                              matrixSamples.shape[1],
                                                              resultVD_Python)

      print("\nPython Naive Output Vector Distance\n")
      print(vd_python)

      print("\n Time Taken by Python Naive Vector Distance in Seconds\n")
      print(time_vd_python)
      
      timeVDSerial.append(time_vd_python)
      #<------------------------------------------------------------------------------------
       
      #----------------- GPU Naive Vector Distance -----------------------------------------> 
      vd_gpu,time_vd_gpu = smote.vectorDistance_gpu(matrixSamples,
                                                    matrixSamples.shape[0],
                                                    matrixSamples.shape[1])

      print("\nGPU Naive Output Vector Distance\n")
      print(vd_gpu)

      print("\n Time Taken by GPU Naive Vector Distance in Seconds\n")
      print(time_vd_gpu)
    
      print("\n\n")
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

      #------------------ Python Naive KNN -------------------------------------->

      resultKnn = np.random.rand(matrixSamples.shape[0]*k,
                                 matrixSamples.shape[1])
      resultKnn = resultKnn.astype(np.float32)
      vectorDistance = vd_gpu.astype(np.float32)   
      
      knn_python, time_knn_python = smote.knn_python(matrixSamples,
                                                     vectorDistance, 
                                                     matrixSamples.shape[0],
                                                     matrixSamples.shape[1],
                                                     k,resultKnn) 

      print("New Points For the Matrix Through KNN Algorithm")
      print(knn_python)
      print("\n")

      print("Time taken by KNN to generate New Matrix in Python")
      print(time_knn_python)
      timeKnnSerial.append(time_knn_python)
      #<----------------------------------------------------------------------------

      #------------------ GPU NAIVE KNN -------------------------------------------->
      resultGpuKnn = np.random.rand(matrixSamples.shape[0]*k,
                                    matrixSamples.shape[1])
      resultGpuKnn = resultGpuKnn.astype(np.float32)
      knn_gpuNaive, time_knn_gpuNaive = smote.knn_gpu(matrixSamples,
                                                      vectorDistance, 
                                                      matrixSamples.shape[0],
                                                      matrixSamples.shape[1],
                                                      k,resultGpuKnn)

      print("\nNew Points For the Matrix Through KNN Algorithm in GPU Naive\n")
      print(knn_gpuNaive)

      print("\n Time Taken by GPU Naive KNN\n")
      print(time_knn_gpuNaive)
      timeKnnGpuNaive.append(time_knn_gpuNaive)
      #<------------------------------------------------------------------------------

      print("\n\n")

       #----------------- Comparision KNN --------------------->
      if(np.allclose(knn_gpuNaive, knn_python, atol=1e-05)):
       
         print("--------------------------------------------------")
         print("----------KNN GPU NAIVE MATCHED-------------------")
         print("--------------------------------------------------")
      else:
         print("KNN NOT MATCHED!")

      #------------------------------------------------------------
      print("\n")
      #------------------ GPU_CPUSORT_ KNN -------------------------------------------->
      resultGpuCpuSortKnn = np.random.rand(matrixSamples.shape[0]*k,
                                    matrixSamples.shape[1])
      resultGpuCpuSortKnn = resultGpuCpuSortKnn.astype(np.float32)
      
      knn_gpu_cpusort, time_knn_gpu_cpusort = smote.knn_gpu_cpusort(matrixSamples,
                                                      vectorDistance,
                                                      matrixSamples.shape[0],
                                                      matrixSamples.shape[1],
                                                      k,resultGpuCpuSortKnn)

      print("\nNew Points For the Matrix Through KNN Algorithm in GPU Naive\n")
      print(knn_gpu_cpusort)
      print("\n Time Taken by GPU  CPU SORT KNN\n")
      print(time_knn_gpu_cpusort)
      timeKnnGpuCpuSort.append(time_knn_gpu_cpusort)
      #<------------------------------------------------------------------------------

      #----------------- Comparision KNN --------------------->
      print("\n\n")
      if(np.allclose(knn_gpu_cpusort, knn_python, atol=1e-05)):
       
         print("-----------------------------------------------------")
         print("----------KNN GPU CPU SORT MATCHED-------------------")
         print("-----------------------------------------------------")
      else:
         print("KNN NOT MATCHED!")

      #------------------------------------------------------------


      print("\n")
      #------------------ GPU_SHARED_ KNN -------------------------------------->
      resultGpuSharedKnn = np.random.rand(matrixSamples.shape[0]*k,
                                    matrixSamples.shape[1])
      resultGpuSharedKnn = resultGpuCpuSortKnn.astype(np.float32)

      knn_gpu_shared, time_knn_gpu_shared = smote.knn_gpu_shared(matrixSamples,
                                                      vectorDistance,
                                                      matrixSamples.shape[0],
                                                      matrixSamples.shape[1],
                                                      k,resultGpuCpuSortKnn)

      print("\nNew Points For the Matrix Through KNN Algorithm in GPU Naive\n")
      print(knn_gpu_shared)
      print("\n Time Taken by GPU  CPU SORT KNN\n")
      print(time_knn_gpu_shared)
      timeKnnGpuShared.append(time_knn_gpu_shared)
      #<------------------------------------------------------------------------------

      #----------------- Comparision KNN --------------------->
      print("\n\n")
      if(np.allclose(knn_gpu_shared, knn_python, atol=1e-05)):

         print("-----------------------------------------------------")
         print("----------KNN GPU SHARED  MATCHED-------------------")
         print("-----------------------------------------------------")
      else:
         print("KNN NOT MATCHED!")

      #------------------------------------------------------------

      print("\n")
      print("New Matrix Appended with Old Matrix")

      matrixSamples=np.append(matrixSamples,knn_python,axis=0)
      print("Shape of Output Matrix =",matrixSamples.shape)

   plt.xlabel("Sizes")
   plt.ylabel("Time taken (ms)")
   plt.yscale('log')
   plt.title("Time taken for generating new matrix for different sizes")
   plt.plot(size,timeVDSerial,label='Vector Distance Python',color="red",marker='o')
   plt.plot(size,timeVDGPU,label='Vector Distance GPU',color="orange",marker='o')
   plt.plot(size,timeKnnSerial,label='Knn Python',color="yellow",marker='o')  
   plt.plot(size,timeKnnGpuNaive,label='Knn Gpu Naive',color="blue",marker='o')
   plt.plot(size,timeKnnGpuCpuSort,label='Knn Gpu with Cpu sort',color="brown",marker='o')
   plt.plot(size,timeKnnGpuShared,label='Knn Gpu with Shared Memory',color="purple",marker='o')
  
   plt.legend()
   plt.savefig('Smote_4.png')
