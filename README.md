# NIDS Attack
Experimental packet modifications intended to evade machine learning based Network Intrusion Detection Systems (NIDSs).

## System Overview
The goal of our system is to modify malicious traffic to make it appear benign to a target Network Intrusion Detection System (NIDS) by perturbing the timestamp and adding no-ops to the header. The important constraint in our project is that our modifications to traffic must not affect their functionality and adhere to netowrk protocols.

![System Model](figures/system_model.png)

## Results
We tested various methods of perturbing the timestamp of malicious traffic with the goal of reducing our targeted NIDS [Whisper's](https://github.com/fuchuanpu/Whisper) detection rate. 

| Attack Type         | Clean Traffic AUC   | Modified Traffic AUC  |
| ------------------- | ------------------- | --------------------- |
| SSL Renegotiation   |             0.63    |               0.46    |
| Syn DoS             |             0.54    |               0.32    |
| Video Injection     |             0.56    |               0.30    |
| ARP MITM            |             0.52    |               0.29    |
| Active Wiretap      |             0.45    |               0.27    |
| OS Scan             |             0.58    |               0.58    |
| SSDP Flood          |             0.87    |               0.87    |


## Datasets
We evaluate our system on datasets from [Kitsune](https://github.com/ymirsky/Kitsune-py). 
The datasets can be found in the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00516).

The script `download_datasets.py` will download the datasets from the repository. 
The script `split_pcaps.py split_datasets_loc num_processes` will split the the full dataset into subsets for training and testing. 20% is used for training while the remaining 80% is used for testing The split datatsets will be written to`split_datasets_loc` in the folders train and test. The number of processes used is defined by `num_processes`.


## Reproducing Results
### Setting up Environment
1. This projects dependencies are listed in `requirements.txt`. You can create a virtual environment with the required dependencies with `python3 pip install -r requirements.txt`. 
2. You will need to download one or more of the datasets listed above by following the download and split instructions. The AUC scores for "clean traffic" were collected on the testing set. The AUC scores for "disguised traffic" were collected on the disguised testing set.

### Training Whisper
1. Following [the paper describing Whisper](https://arxiv.org/pdf/2106.14707.pdf), we use only benign traffic from the first 20% of each traffic dataset to find cluster centroids. Training entails two steps. 
  a) **Find embeddings for the benign traffic:** `python3 get_embeddings <train_dataset_path> <results_path> <num_processes>`. `train_dataset_path` is a directory that contains a subdirectory for each traffic dataset. The program will write the resulting embeddings as `.npz` files to the `results_path` directory. The program uses the multiprocessessing model to process each dataset in parallel. The number of processes used is defined by `num_processes`. Run `python3 get_embeddings -h` for more help with command line arguments.
  b) **Find centroids for the benign traffic for each attack type:** `python3 k_means_learner.py <embedding_dir_loc> <num_clusters> <save_loc>`. `embedding_dir_loc` should be the same path used in `results_path` of the previous step. We used 10 as the number of clusters in our experiments. The program will save the centroids to a directory called `save_loc` in which each traffic dataset will have a json file filled with centroids.
  
### Evaluating Whisper
1. Given centroids for each attack type, we can play back testing datasets to determine Whisper's accuracy. To run Whisper on each testing dataset, use the following command: `python3 run_detection.py <datasets_loc> <clusters_loc> <results_loc> <num_processes>`. 
  - `datasets_loc` is a directory which has a subdirectory for each testing traffic dataset.
  - `clusters_loc` is a directory which has a json file full of centroids for each attack type. 
  - The program will create a directory called `results_loc` that will contain csv files with a column for expected label, and a column for centroid distance. 
  - The program will run `num_processess` in parallel. Each process will run detection on a different dataset.
2. Given result .csv files, we can plot the ROC curves and find each curve's AUC by running `python3 plot_roc.py <results_loc> <save_loc> <plot_title>`. 
  - `results_loc` is the same `results_loc` that contains csv files with distance and expected label.
  - `save_loc` is a path for where to save the ROC plot.
  - `plot_title` is what to name the ROC curve. 

### Disguising Traffic
1. To disguise the traffic in the testing set by ONLY modifying the header length, you can run `python3 benchmark_disguiser.py <datasets_loc> <centroid_loc> <results_loc> <num_processes>`. 
  - `datasets_loc` is a directory that contains a subdirectory for each type of traffic in the testing set. 
  - `centroid_loc` is a directory that contains a json file full of centroids for each attack type. 
  - The program will write a heirarchy mirroring `datasets_loc` at `results_loc`. However, the .pcapng files at `results_loc` will be "disguised."
  - The program will run `num_processes` in parallel. Each process works on a different dataset. 
2. The disguising traffic process detailed in step 1 can be repeated using `python3 random_and_header_disguiser.py <datasets_loc> <centroid_loc> <results_loc> <num_processes>` to disguise the traffic with a random timestamp perturbation and header modification.
3. The disguising traffic process detailed in step 1 can be repeated using `python3 random_only_disguiser.py <datasets_loc> <centroid_loc> <results_loc> <num_processes>` to disguise the traffic with only a random timestamp perturbation.
4. The disguising traffic process detailed in step 1 can be repeated using `python3 fixed_disguiser.py <datasets_loc> <centroid_loc> <results_loc> <num_processes>` to disguise the traffic with a fxed timestamp perturbation and header modification.
5. The disguising traffic process detailed in step 1 can be repeated using `python3 fixed_disguiser.py <datasets_loc> <centroid_loc> <results_loc> <num_processes>` to disguise the traffic with a loss-informed timestamp perturbation and header modification.

### Evaluating Whisper on Disguised Traffic
The process to evaluate Whisper on disguised traffic is largely the same as the process for evaluating Whisper on the regular testing set. Instead of using the directory of the testing set for `datasets_loc`, use the directory generated by running the disguiser. 
