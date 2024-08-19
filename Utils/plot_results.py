import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('../Results/training_results_MNIST100.csv')
print(df.head())

# Plot accuracy
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(df['Epoch'], df['Train Accuracy'], label='Training Accuracy', marker='o')
plt.plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

# Plot F1-Score
plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(df['Epoch'], df['Train F1-Score'], label='Training F1-Score', marker='o')
plt.plot(df['Epoch'], df['Test F1-Score'], label='Test F1-Score', marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('F1-Score over Epochs')
plt.legend()

plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd subplot
plt.plot(df['Epoch'], df['Train Loss'], label='Training loss', marker='o')
plt.plot(df['Epoch'], df['Test Loss'], label='Test loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('F1-Score over Epochs')
plt.legend()

# Add a title for the whole plot
plt.suptitle('MNIST', fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the space to fit the suptitle
plt.show()
