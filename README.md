# Parallelising a CNN code using Horovod and GPUs

## Repository for the medium article.

* The processed data is included as a tarball. 

* The CNN codes are in the folder *python_code*
  * *cnn.py*: serial CNN code
  * *horovod_dist_gen.py*: data parallelism using TensorFlow image data generator
  * *horovod_dist_shard.py*: data parallelism using sharding
* The slurm scripts for running the Python codes on a HPC infrastructure are at the root level
  * *run_cnn.py*: to run the serial CNN code
  * *run_cnn_horovod.py*: to run the parallelised code (either *horovod_dist_gen.py* or *horovod_dist_shard.py*)
* The *plots* folder has some loss and accuracy curves for reference. The first integer in the filename corresponds to the number of GPUs used for that run.