# SMOTE_GPU

1) Optimized VectorDistance with GPU
2) Including VectorDistance with KNN, solved class imbalance problem using SMOTE tenchnique

Included the report + ppt explaining the algorithm for parallelization.

The respective folders have the code, which includes

1) The python code file
2) TimeGraph to compare time difference between CPU and GPU (using different methods)
3) To see the GPU utilization, observe the Profiling images
4) Included the output.txt as well, which includes the traces of output. 

Please note, these experiments were performed on Google Cloud Platform, using a machine with the following configuration:

➢ Machine type: n1-standard-4 (4 vCPUs, 15 GB memory)

➢ CPU platform: Intel Haswell

➢ GPUs: 1 x NVIDIA Tesla T4

➢ OS: ubuntu-1804-bionic-v20210825
