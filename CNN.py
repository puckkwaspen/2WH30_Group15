### This is our first attempt at a CNN

if __name__ == '__main__':
    import torch
    import torchvision
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data import random_split
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os
    import pandas as pd
    from torch.utils.data import Dataset
    from PIL import Image
    from data_preparation import MaterialDataset, binary_image_label_mapping, image_dir, train_transform, val_transform

    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)


    # Initialize dataset
    dataset = MaterialDataset(image_dir=image_dir, label_mapping=binary_image_label_mapping)

    # Define percentages for train, validation, and test splits
    train_percentage = 0.8
    test_percentage = 0.2

    # Calculate lengths for each split
    train_size = int(train_percentage * len(dataset))
    test_size = len(dataset) - train_size  # Remaining for the test set

    # Split the dataset
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Print lengths of splits
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Test Data : {len(test_data)}")

    # Assign specific transforms for training and validation datasets
    train_data.dataset.transform = train_transform
    test_data.dataset.transform = val_transform

    # Define batch size
    batch_size = 10

    # Create data loaders with batching
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    # Print a message indicating the loader sizes
    print(f"Train DataLoader contains {len(train_dl)} batches")
    print(f"Test DataLoader contains {len(test_dl)} batches")


    def show_batch(dl):
        """Plot images grid of single batch"""
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break


    show_batch(train_dl)

    # class ImageClassificationBase(nn.Module):
    #
    #     def training_step(self, batch):
    #         images, labels = batch
    #         out = self(images)  # Generate predictions
    #         loss = F.cross_entropy(out, labels)  # Calculate loss
    #         return loss
    #
    #     def validation_step(self, batch):
    #         images, labels = batch
    #         out = self(images)  # Generate predictions
    #         loss = F.cross_entropy(out, labels)  # Calculate loss
    #         acc = accuracy(out, labels)  # Calculate accuracy
    #         return {'val_loss': loss.detach(), 'val_acc': acc}
    #
    #     def validation_epoch_end(self, outputs):
    #         batch_losses = [x['val_loss'] for x in outputs]
    #         epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    #         batch_accs = [x['val_acc'] for x in outputs]
    #         epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    #         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    #
    #     def epoch_end(self, epoch, result):
    #         print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
    #             epoch, result['train_loss'], result['val_loss'], result['val_acc']))


    # Define a simple CNN for binary classification
    class MaterialClassificationCNN(nn.Module):
        def __init__(self):
            super(MaterialClassificationCNN, self).__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Flatten(),
                nn.Linear(128 * 18 * 18, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()  # For binary classification
            )

        def forward(self, xb):
            return self.network(xb)


    def precision_and_recall(outputs, labels):
        preds = (outputs > 0.5).float()
        tp = torch.sum((preds == 1) & (labels == 1)).item()  # True Positives
        fp = torch.sum((preds == 1) & (labels == 0)).item()  # False Positives
        fn = torch.sum((preds == 0) & (labels == 1)).item()  # False Negatives

        precision = tp / (tp + fp + 1e-8)  # Add small epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)

        return precision, recall


    def fbeta_score(outputs, labels, beta=0.5):
        precision, recall = precision_and_recall(outputs, labels)
        beta_squared = beta ** 2
        fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + 1e-8)
        return fbeta

    def accuracy(outputs, labels):
        preds = (outputs > 0.5).float()
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    @torch.no_grad()
    #def evaluate(model, test_loader):
       # model.eval()
       # test_losses, test_accs = [], []
       # for images, labels in test_loader:
           # images, labels = images.to(device), labels.to(device)
           # outputs = model(images)
           # loss = F.binary_cross_entropy(outputs, labels.unsqueeze(1))
           # acc = accuracy(outputs, labels.unsqueeze(1))
           # test_losses.append(loss.item())
           # test_accs.append(acc.item())
        #return {'loss': sum(test_losses) / len(test_losses), 'accuracy': sum(test_accs) / len(test_accs)}


    @torch.no_grad()
    def evaluate(model, test_loader):
        model.eval()
        test_losses, test_accs, test_fbeta = [], [], [] # adding a new list
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.binary_cross_entropy(outputs, labels.unsqueeze(1))
            acc = accuracy(outputs, labels.unsqueeze(1))
            fbeta = fbeta_score(outputs, labels.unsqueeze(1), beta=0.5) # using the f0.5 score
            test_losses.append(loss.item())
            test_accs.append(acc.item())
            test_fbeta.append(fbeta)
        return {
            'loss': sum(test_losses) / len(test_losses),
            'accuracy': sum(test_accs) / len(test_accs),
            'f0.5': sum(test_fbeta) / len(test_fbeta)
        }



    def fit(epochs, lr, model, train_loader, test_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):

            model.train()
            train_losses = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.binary_cross_entropy(outputs, labels.unsqueeze(1))
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = evaluate(model, test_loader)
            result['train_loss'] = sum(train_losses) / len(train_losses)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {result['train_loss']:.4f}, "
                  f"Test Loss: {result['loss']:.4f}, Accuracy: {result['accuracy']:.4f}, "
                  f"F0.5: {result['f0.5']:.4f}") # Adding the f0.5 to the print statement
            history.append(result)
        return history


    # Initialize and train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaterialClassificationCNN().to(device)

    history = fit(epochs=2, lr=0.001, model=model, train_loader=train_dl, test_loader=test_dl)

    def plot_accuracies(history):
        """ Plot the history of accuracies"""
        accuracies = [x['accuracy'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs');


    plot_accuracies(history)


    def plot_losses(history):
        """ Plot the losses in each epoch"""
        train_losses = [x.get('train_loss') for x in history]
        test_losses = [x['loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(test_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs');


    plot_losses(history)


    def plot_fbeta(history):
        """Plot the history of F0.5 scores"""
        fbetas = [x['f0.5'] for x in history]
        plt.plot(fbetas, '-gx')
        plt.xlabel('epoch')
        plt.ylabel('F0.5')
        plt.title('F0.5 Score vs. No. of epochs');


    plot_fbeta(history)
