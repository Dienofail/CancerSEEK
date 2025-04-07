#!/bin/bash

# Script to run the synthetic feature oversampling experiment
# Adds 3 synthetic features with alpha=0.8 and orthogonality=0.5

echo "Running synthetic_feature_oversample.py..."
echo "Parameters:"
echo "  Num Synthetic Features: 3"
echo "  Target Alpha: 0.8"
echo "  Orthogonality: 0.5"
echo "  Standardization: Yes"
echo "  Log Transform: None"
echo "  Outer CV Folds: 5"
echo "  Inner CV Folds: 3"
echo "  Random Seed: 42"
echo "--------------------------------------"

# Execute the Python script
python /Users/alvinshi/Library/CloudStorage/Dropbox/Interview_prep/Exact_Sciences/CancerSEEK/synthetic_feature_oversample.py \
    --num-synthetic-features 10 \
    --alpha 0.6 \
    --orthogonality 0.4 \
    --standardize \
    --log-transform none \
    --outer-splits 5 \
    --inner-splits 3 \
    --random-seed 42 \
    --target-specificity 0.99

EXIT_CODE=$?
echo "--------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Script finished successfully."
else
    echo "Script failed with exit code $EXIT_CODE."
fi 