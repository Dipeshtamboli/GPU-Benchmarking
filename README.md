# GPU-Benchmarking

This code is for benchmarking the GPU performance by running experiments on the different deep learning architectures. The code is inspired from the [pytorch-gpu-benchmark](https://github.com/ryujaehun/pytorch-gpu-benchmark) repository.

The code uses PyTorch deep models for the evaluation. It considers three different precisions for training and inference. In training, back-propagation is included. 

Double precision: 64 bits           
Float (single) precision: 32 bits    
Half precision: 16 bits 


## The has following components:
- `torch_train_gpu.py` code for running the experiments and collecting the time taken for each iteration in a CSV file
- `benchmark.sh` is a bash script for checking the available GPUs and running `torch_train_gpu.py` for each GPU.
-  `data_csv_to_json.py` converts all the CSV data to a dictionary of the dictionary format
- `plot_json.py` for plots corresponding to different servers as well as averaging over different deep models.
- `*_results` folder contains the CSV files corresponding to each precision and mode. Also, it contains more information about the GPUs. (cores, model)
    - [Gilbreth info](https://github.com/Dipeshtamboli/GPU-Benchmarking/blob/master/gilbreth_results/system_info.txt)
    - [Gilbreth-G sub-cluster info](https://github.com/Dipeshtamboli/GPU-Benchmarking/blob/master/gilbrethG_results/system_info.txt)

    - [Bell info](https://github.com/Dipeshtamboli/GPU-Benchmarking/blob/master/bell_results/system_info.txt)

    - [Draco info](https://github.com/Dipeshtamboli/GPU-Benchmarking/blob/master/draco_results/system_info.txt)

    - [Auriga info](https://github.com/Dipeshtamboli/GPU-Benchmarking/blob/master/auriga_results/system_info.txt)
- `server_wise_plots` folder contains individual server plots comparing the time taken for different deep models for different precisions and modes (training/inference).

![Result image](https://github.com/Dipeshtamboli/GPU-Benchmarking/blob/master/time_taken%20vs%20experiment_modes%20for%20different%20GPUs.png?raw=true)

This plot shows the average time taken by each GPU for different precisions and modes (training/inference). This is average over different deep learning architectures. From here, we can deduce that GilbrethG is performing better than Bell.

