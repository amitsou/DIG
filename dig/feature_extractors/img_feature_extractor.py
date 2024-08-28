import argparse

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def parse_args():
    """Parse the command line arguments
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Input path directory containing the dynamic images"
    )
    parser.add_argument(
        "-p", "--print", help="Flag to print messages", action="store_true"
    )
    parser.add_argument("-m", "--model", help="Model to use for extracting features")
    return parser.parse_args()


def main():
    # TODO: Select more models

    # VGG_TYPES =
    vgg19 = models.vgg19(pretrained=True)

    # TODO: Load the image
    # img = Image.open("0BDCZEGO.jpg")
    img = Image.open("2_2_read.jpg")

    # TODO: Apply transformations to the image
    # These transforms refer to the VGG architecture (224x224)
    # In case of using another architecture, the transforms must be adjusted accordingly
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = transform(img)

    # Add a batch dimension to the image
    img = img.unsqueeze(0)

    # Set the model to evaluation mode
    vgg19.eval()

    # Pass the image through the model
    with torch.no_grad():
        # Forward pass the image through VGG19
        output = vgg19.features(img)

        # Get the second-to-last dense layer (index -2)
        second_to_last_layer_output = vgg19.classifier[-2](
            output.view(output.size(0), -1)
        )

    # Convert the feature vector to a numpy array
    feature_vector = second_to_last_layer_output.numpy()

    # Print the feature vector
    print("Feature Vector:")
    print(feature_vector)

    # Save the feature vector as a numpy file
    # np.save("feature_vector.npy", feature_vector)


if __name__ == "__main__":
    main()
