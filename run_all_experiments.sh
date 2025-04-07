#!/bin/bash

# Arrays of parameters to iterate through
DETECTION_MODELS=("LR" "XGB" "TF" "MOE")
LOCALIZATION_MODELS=("RF" "XGB" "TF" "MOE")
TARGET_SPECS=(985 99 994)

# Total combinations for progress tracking
TOTAL_COMBINATIONS=$((${#DETECTION_MODELS[@]} * ${#LOCALIZATION_MODELS[@]} * ${#TARGET_SPECS[@]}))
CURRENT=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

echo "Starting runs for all combinations (total: $TOTAL_COMBINATIONS)"

# Create a log directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for detection_model in "${DETECTION_MODELS[@]}"; do
  for localization_model in "${LOCALIZATION_MODELS[@]}"; do
    for target_spec in "${TARGET_SPECS[@]}"; do
      CURRENT=$((CURRENT + 1))
      
      # Log file for this run
      LOG_FILE="logs/run_${detection_model}_${localization_model}_${target_spec}.log"
      
      echo "[$CURRENT/$TOTAL_COMBINATIONS] Running with detection-model=$detection_model, localization-model=$localization_model, target-spec=$target_spec"
      
      # Record start time in the log
      echo "Run started at: $(date)" > "$LOG_FILE"
      echo "Parameters: detection-model=$detection_model, localization-model=$localization_model, target-spec=$target_spec" >> "$LOG_FILE"
      echo "-------------------------------------------------" >> "$LOG_FILE"
      
      # Run the Python script with the current parameters
      python paper_reproduction.py --detection-model "$detection_model" --localization-model "$localization_model" --target-spec "$target_spec" >> "$LOG_FILE" 2>&1
      
      if [ $? -eq 0 ]; then
        echo "  Completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      else
        echo "  Failed: See $LOG_FILE for details"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
      fi
    done
  done
done

echo "All runs completed."
echo "Summary: $SUCCESS_COUNT successful, $FAILURE_COUNT failed (out of $TOTAL_COMBINATIONS)"
echo "Results are stored in the logs directory."
