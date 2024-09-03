#!/bin/bash

# Get the current working directory (base directory)
base_dir=$(pwd)

# Array of dataset directories and their corresponding scripts
datasets=("DREAM5_net1_FGRN:DeepFGRN_DREAM5net1_FCV.py"
          "Ecoli_cold_FGRN:DeepFGRN_cold_FCV.py"
          "Ecoli_heat_FGRN:DeepFGRN_heat_FCV.py"
          "Ecoli_lactose_FGRN:DeepFGRN_lactose_FCV.py"
          "Ecoli_oxidative_FGRN:DeepFGRN_oxidative_FCV.py"
          "human_breast_FGRN:DeepFGRN_Breast_FCV.py"
          "human_COVID_FGRN:DeepFGRN_COVID_FCV.py"
          "human_liver_FGRN:DeepFGRN_Liver_FCV.py"
          "human_lung_FGRN:DeepFGRN_Lung_FCV.py")

# Loop through each dataset directory
for dataset_script in "${datasets[@]}"; do
    # Split the string into dataset directory and script name
    IFS=":" read -r dataset script <<< "$dataset_script"

    echo "Processing $dataset..."
    
    # Navigate to the dataset directory
    cd "$base_dir/$dataset"
    
    # Run the corresponding python script
    python "$script"
    
    # Navigate back to the base directory
    cd "$base_dir"
done

echo "All datasets have been processed."
