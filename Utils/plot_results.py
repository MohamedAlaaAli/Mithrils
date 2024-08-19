import pandas as pd
import matplotlib.pyplot as plt


def visualize_results(csv_path:str, title:str):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(df.head())

    # Plot accuracy
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(df['Epoch'].values, df['Train Accuracy'].values, label='Training Accuracy', marker='o')
    plt.plot(df['Epoch'].values, df['Test Accuracy'].values, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Plot F1-Score
    plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(df['Epoch'].values, df['Train F1-Score'].values, label='Training F1-Score', marker='o')
    plt.plot(df['Epoch'].values, df['Test F1-Score'].values, label='Test F1-Score', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('F1-Score over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd subplot
    plt.plot(df['Epoch'].values, df['Train Loss'].values, label='Training loss', marker='o')
    plt.plot(df['Epoch'].values, df['Test Loss'].values, label='Test loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Loss over Epochs')
    plt.legend()

    # Add a title for the whole plot
    plt.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the space to fit the suptitle
    plt.show()


if __name__ == '__main__':
    csv_pth = input("Enter csv path: ")
    title = input("Enter dataset name: ")
    visualize_results(csv_pth, title)
    
    