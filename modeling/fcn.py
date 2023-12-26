import torch
from torchvision.models.segmentation import fcn_resnet101

class FCNResNet101(torch.nn.Module):
    def __init__(self, num_classes,size):
        super(FCNResNet101, self).__init__()
        # Load the pre-trained fcn_resnet101 model
        self.model = fcn_resnet101(pretrained=True)
        
        # Replace the final classifier to match the number of classes
        self.model.classifier[4] = torch.nn.Conv2d(size, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)

# Example usage:
if __name__ == "__main__":
    num_classes = 2  # Change to the number of classes in your segmentation task
    size = 512
    model = FCNResNet101(num_classes,size)
    input = torch.randn(1, 3, size, size)  # Change the input size as needed
    output = model(input)
    print(output['out'].size())