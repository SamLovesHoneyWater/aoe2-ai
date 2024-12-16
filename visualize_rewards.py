from process_rewards import *
from utils import list_folders
from actions import DataStream

# List all data
directory_path = 'data/'
folders = list_folders(directory_path)
print(f"Folders in '{directory_path}': {folders}")
# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the raw scores and smoothed scores
for folder in folders:
    try:
        data = DataStream(filename=directory_path + folder, load=True)
    except FileNotFoundError:
        continue
    scores = [datapoint['score'] for datapoint in data.stream]
    smooth = process_rewards(scores)
    
    ax1.plot(scores, label=folder)
    ax2.plot(smooth, label=folder)

# Set labels and titles for the raw data subplot
ax1.set_xlabel('Datapoint Index')
ax1.set_ylabel('Score')
ax1.set_title('Raw Scores of Every Run')
ax1.legend()

# Set labels and titles for the smoothed data subplot
ax2.set_xlabel('Datapoint Index')
ax2.set_ylabel('Score')
ax2.set_title('Smoothed Scores of Every Run')
ax2.legend()

# Set the same y-axis limits for both subplots
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# Set the same x-axis limits for both subplots
x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
ax1.set_xlim(x_min, x_max)
ax2.set_xlim(x_min, x_max)

# Show the plots
plt.tight_layout()
plt.show()
