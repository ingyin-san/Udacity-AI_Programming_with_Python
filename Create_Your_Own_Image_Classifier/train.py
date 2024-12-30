import argparse
import torch
import json
from torchvision import transforms, datasets, models
from torch import nn, optim

def train(args):
    #Define the transforms for the training, validation, and testing sets
    training_transform = transforms.Compose([transforms.RandomRotation(45),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    validation_transform = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    testing_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    #Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(args.data_directory + '/train', transform=training_transform)
    valid_dataset = datasets.ImageFolder(args.data_directory + '/valid', transform=validation_transform)
    test_dataset = datasets.ImageFolder(args.data_directory + '/test', transform=testing_transform)

    #Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    #Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    #Load a pretrained network
    model = getattr(models, args.arch)(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False

    #Define a new, untrained feed-forward network
    model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(args.hidden_units, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(1024, 102),
                                     nn.LogSoftmax(dim = 1)
                                     )
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    #Training the network
    model.to(args.device)
    
    steps = 0
    print_every = 50

    for epoch in range(args.epochs):
        train_loss = 0
        model.train()

        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()

            logps = model(inputs)

            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(args.device), labels.to(args.device)

                        logps = model(inputs)
                        loss = criterion(logps, labels)
                        validation_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Train loss: {train_loss/print_every:.3f}, "
                    f"Validation loss: {validation_loss/len(valid_loader):.3f}, "
                    f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                
                train_loss = 0
                model.train()

    #Save the checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'arch': args.arch
    }

    torch.save(checkpoint, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, help="data directory")
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help="path to save checkpoint")
    parser.add_argument('--arch', type=str, default='vgg16', help="model architecture")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--hidden_units', type=int, default=512, help="number of hidden units")
    parser.add_argument('--epochs', type=int, default=7, help="number of epochs")
    parser.add_argument('--gpu', action='store_true', help="use GPU if available")
    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    train(args)
