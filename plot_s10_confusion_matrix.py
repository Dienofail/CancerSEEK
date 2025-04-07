import matplotlib.pyplot as plt
from process_supplemental_data import load_s10_confusion_matrix
from plotting import plot_tissue_localization_confusion_matrix

# Define the output path for the plot
output_pdf = "s10_confusion_matrix_plot.pdf"

# Load the confusion matrix data
print("Loading S10 confusion matrix data...")
cm_df_raw = load_s10_confusion_matrix()
print("Data loaded.")

# The original table S10 has rows = predicted, columns = actual
# The plotting function expects rows = actual (true), columns = predicted
# Therefore, we need to transpose the DataFrame
print("Transposing the confusion matrix...")
cm_df_transposed = cm_df_raw.T 
print("Matrix transposed.")

# Get the cancer types from the transposed matrix index (which are the true labels)
cancer_types = list(cm_df_transposed.index)

print(f"Plotting confusion matrix for types: {cancer_types}")

# Plot the confusion matrix using the pre-computed DataFrame
# The data is already in fractions (0-1), so no normalization needed for calculation.
# The plotting function will format it appropriately.
fig, ax, _ = plot_tissue_localization_confusion_matrix(
    cm=cm_df_transposed,
    cancer_types=cancer_types,
    figsize=(10, 8),
    localization_model="Original Study", # Specify model/source if desired
    save_path=output_pdf # Specify the output file path
)

# Close the plot figure to prevent it from displaying interactively if not needed
plt.close(fig)

print(f"Confusion matrix plot saved to {output_pdf}") 