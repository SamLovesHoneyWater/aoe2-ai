import torch
import torch.nn as nn

class GameAI(nn.Module):
    def __init__(self, input_channels=1, input_height=1080, input_width=1920):
        super(GameAI, self).__init__()
        
        # Convolutional layers to process the screenshot
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output((input_channels, input_height, input_width))
        
        # Fully connected layers for different outputs
        self.fc_shared = nn.Linear(self.feature_size, 512)
        
        # Specific output layers
        self.onehot_head = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
        self.continuous_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.discrete_head1 = nn.Linear(512, 3)  # For -1, 0, 1
        self.discrete_head2 = nn.Linear(512, 3)  # For -1, 0, 1

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        # Process the screenshot through conv layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        # Shared features
        shared_features = torch.relu(self.fc_shared(x))
        
        # Generate different outputs
        onehot_output = self.onehot_head(shared_features)
        continuous_output = self.continuous_head(shared_features)
        discrete1_output = self.discrete_head1(shared_features)
        discrete2_output = self.discrete_head2(shared_features)
        
        return {
            'onehot': onehot_output,
            'continuous': continuous_output,
            'discrete1': discrete1_output,
            'discrete2': discrete2_output
        }

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = GameAI()
    model = model.to(device)
    
    # Example input (batch size = 1)
    sample_input = torch.randn(1, 1, 1080, 1920).to(device)
    
    # Get outputs
    outputs = model(sample_input)
    print(f"Using device: {device}")
    print("Output shapes:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
        print(value)
