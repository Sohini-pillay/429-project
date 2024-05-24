#!/bin/sh 
# Usage: ./run_everything.sh <unique_tag> <dataset_loc>
python3 src/get_embeddings.py $2 embeddings_$1 3 
python3 src/k_means_learner.py embeddings_$1 10 centroid_$1 
python3 src/run_detection.py $2 centroid_$1 detection_results_$1 3
python3 src/plot_roc.py detection_results_$1 roc_$1.png $1
