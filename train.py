'''
# Project Name

Copyright © 2023 Chengjui Fan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Works Cited:
This project uses the "Perplexity Labs, 70b Model" for some of its functionalities. The model was accessed on Nov 25, 2023. labs.perplexity.com

'''
import torch
from torch import nn
from torchvision import models
import torchaudio
class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()

        # Word Embedding
        self.word_embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)

        # Image Embedding
        self.image_embedding = models.resnet50(pretrained=True)
        self.image_embedding.fc = nn.Linear(self.image_embedding.fc.in_features, 256)

        # Audio Embedding
        self.audio_embedding = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=256,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23, 'center': False}
        )

        # Shared Architecture Functions
        self.shared_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Output Layer
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x_word, x_image, x_audio):
        x_word = self.word_embedding(x_word)
        x_image = self.image_embedding(x_image)
        x_audio = self.audio_embedding(x_audio)

        # Concatenate the embeddings
        x = torch.cat((x_word, x_image, x_audio), dim=1)

        x = self.shared_layers(x)
        x = self.output_layer(x)

        return x

def load_dataset(self, directory, labels):
    # Initialize lists for different types of files
    tokenized_text = []
    image_embeddings = []
    audio_embeddings = []
    target_labels = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check the file type and process accordingly
        if filename.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()
            tokenized_text.append(self.tokenizer(text))
        elif filename.endswith('.jpg') or filename.endswith('.png'):
            image_embeddings.append(self.image_embedding(file_path))
        elif filename.endswith('.wav') or filename.endswith('.mp3'):
            audio_embeddings.append(self.audio_embedding(file_path))

        # Get the label for this instance
        label = labels[filename]
        target_labels.append(label)

    # Create the embeddings
    word_embeddings = torch.zeros(len(tokenized_text), self.embedding_dim)
    for i, tokens in enumerate(tokenized_text):
        word_embeddings[i, :] = self.word_embedding(tokens)

    # Convert labels to multi-label format
    target_labels = convert_to_multilabel(target_labels)

    # Return the embeddings and labels
    return word_embeddings, image_embeddings, audio_embeddings, target_labels

def convert_to_multilabel(labels, all_possible_labels):
    # Initialize a zero matrix with shape (number of instances, number of classes)
    multi_labels = np.zeros((len(labels), len(all_possible_labels)))

    # Iterate over labels
    for i, label in enumerate(labels):
        # Split the labels by comma
        instance_labels = label.split(',')

        # For each label, find its index in all_possible_labels and set the corresponding element to 1
        for instance_label in instance_labels:
            index = all_possible_labels.index(instance_label)
            multi_labels[i, index] = 1

    return multi_labels


# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # This loss function is suitable for multi-label classification
optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for (text_data, image_data, audio_data), targets in dataloader:
        # Move the data and targets to the device
        text_data = text_data.to(device)
        image_data = image_data.to(device)
        audio_data = audio_data.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model.forward(text_data, image_data, audio_data)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the running loss
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
