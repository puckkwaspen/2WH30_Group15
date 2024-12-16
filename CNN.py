### Binary classification of plastic vs non-plastic materials with CNN

if __name__ == '__main__':
    import torch
    import torchvision
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data import random_split
    from torchvision.utils import make_grid
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os
    import pandas as pd
    from torch.utils.data import Dataset
    from PIL import Image

    from data_preparation import MaterialDataset, binary_image_label_mapping, image_dir, train_transform, val_transform
    from itertools import product
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    # Set the seed for reproducibility
    seed = 987
    torch.manual_seed(seed)
    random.seed(seed)


    # Define transformations for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # val_transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

    # Initialize dataset
    dataset = MaterialDataset(image_dir=image_dir, label_mapping=binary_image_label_mapping)

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create training and validation datasets with transformations
    train_dataset = MaterialDataset(
        image_dir=image_dir,
        label_mapping=binary_image_label_mapping,
        transform=train_transform  # Full transformation for training
    )

    val_dataset = MaterialDataset(
        image_dir=image_dir,
        label_mapping=binary_image_label_mapping,
        transform=test_transform  # Minimal transformation for testing
    )

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=seed)

    # Create subsets for training and testing
    train_data = Subset(train_dataset, train_indices)
    test_data = Subset(val_dataset, test_indices)
    # Split the dataset 80-20


# The commented code is just for visualization, not needed right now
    # def show_batch(dl):
    #     """Plot images grid of single batch"""
    #     for images, labels in dl:
    #         fig, ax = plt.subplots(figsize=(16, 12))
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    #         break
    #
    #
    # show_batch(train_dl)

    # Define a simple CNN for binary classification
    # Changing the CNN architecture so that experiments can be run
    # Architecture Search Configuration
    architecture_space = {
        'num_conv_layers': [2, 3, 4],  # Vary number of conv layers
        'filters': [16, 32, 64],  # Number of filters in conv layers
        'kernel_sizes': [3, 5],  # Kernel size options
        'use_batch_norm': [True, False],  # Batch norm inclusion
        'dropout_rate': [0.2, 0.4, 0.5],  # Dropout options
        'pooling_type': ['max', 'avg'],  # Pooling types
        'num_fc_layers': [1, 2],  # Number of fully connected layers
        'fc_units': [128, 256],  # Units in FC layers
    }


    # Define a Configurable CNN with Dynamic Architecture
    class ConfigurableCNN(nn.Module):
        def __init__(self, arch_config):
            super(ConfigurableCNN, self).__init__()
            self.layers = nn.ModuleList()
            in_channels = 3

            # Convolutional layers
            for i in range(arch_config['num_conv_layers']):
                out_channels = arch_config['filters']
                kernel_size = arch_config['kernel_sizes']
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
                if arch_config['use_batch_norm']:
                    self.layers.append(nn.BatchNorm2d(out_channels))
                self.layers.append(nn.ReLU())
                pooling = nn.MaxPool2d(2) if arch_config['pooling_type'] == 'max' else nn.AvgPool2d(2)
                self.layers.append(pooling)
                in_channels = out_channels

            # Adaptive Pooling
            self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))

            # Fully connected layers
            fc_layers = []
            input_dim = arch_config['filters']
            for _ in range(arch_config['num_fc_layers']):
                fc_layers.append(nn.Linear(input_dim, arch_config['fc_units']))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(arch_config['dropout_rate']))
                input_dim = arch_config['fc_units']
            fc_layers.append(nn.Linear(input_dim, 1))  # Binary output
            self.fc = nn.Sequential(*fc_layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            x = torch.flatten(x, 1)
            return torch.sigmoid(self.fc(x))



    # Evaluate Configurations with Cross-Validation
    def cross_validate_model(model_class, train_data, arch_config, k=5, epochs=10, lr=0.001, batch_size=32,
                             optimizer=torch.optim.Adam, weight_decay=0.0):
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            print(f"Fold {fold + 1}/{k}")

            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(train_data, batch_size=batch_size, sampler=val_sampler)

            # Initialize model using the lambda function and pass the same arch_config for all folds
            model = model_class(arch_config=arch_config).to(device)
            opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.BCELoss()

            # Training Loop
            for epoch in range(epochs):
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # Validation
            model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(images)
                    val_loss += loss_fn(outputs, labels).item()
                    val_acc += accuracy(outputs, labels)

            results.append({'loss': val_loss / len(val_loader), 'accuracy': val_acc / len(val_loader)})

        return results

        # Uncomment this and comment lines 119-123 if we want to use 224x224
        #     self.flattened_size = 128 * 28 * 28
        #
        #     self.fc = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(self.flattened_size, 128),  # Match input size here
        #         nn.ReLU(),
        #         nn.Linear(128, 1),
        #         nn.Sigmoid()  # Binary classification
        #     )




# If we want to use 224x224 images
        # def forward(self, xb):
        #     xb = self.network(xb)
        #     return self.fc(xb)

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
                  f"F0.5: {result['f0.5']:.4f}")  # Adding the f0.5 to the print statement
            history.append(result)
        return history

#Implementing k-fold CV

    def random_search_with_cv(train_data, param_grid, architecture_space, n_iter=10, k=5):
        keys, values = zip(*param_grid.items())
        param_combinations = list(product(*values))
        random.shuffle(param_combinations)
        param_combinations = param_combinations[:min(n_iter, len(param_combinations))]

        results = []
        best_performance = float('-inf')
        best_params = None

        for i, params in enumerate(param_combinations):
            print(f"Iteration {i + 1}/{n_iter} with params: {params}")

            # Unpack hyperparameters
            param_dict = dict(zip(keys, params))
            batch_size = param_dict['batch_size']
            lr = param_dict['lr']
            epochs = param_dict['epochs']
            optimizer = param_dict['optimizer']
            dropout_rate = param_dict['dropout_rate']
            pooling_after_conv = param_dict['pooling_after_conv']
            weight_decay = param_dict['weight_decay']

            # Create architecture configuration
            arch_config = {
                'num_conv_layers': random.choice(architecture_space['num_conv_layers']),
                'filters': random.choice(architecture_space['filters']),
                'kernel_sizes': random.choice(architecture_space['kernel_sizes']),
                'use_batch_norm': random.choice(architecture_space['use_batch_norm']),
                'dropout_rate': dropout_rate,
                'pooling_type': random.choice(architecture_space['pooling_type']),
                'num_fc_layers': random.choice(architecture_space['num_fc_layers']),
                'fc_units': random.choice(architecture_space['fc_units']),
            }

            print(f"Testing architecture configuration: {arch_config}")

            # Perform cross-validation with combined params
            fold_results = cross_validate_model(
                model_class=lambda arch_config=arch_config: ConfigurableCNN(arch_config),
                train_data=train_data,
                arch_config=arch_config,
                k=k,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                optimizer=optimizer,
                weight_decay=weight_decay
            )

            # Calculate mean accuracy across folds
            mean_accuracy = sum(f['accuracy'] for f in fold_results) / len(fold_results)

            # Store results and track the best configuration
            results.append((param_dict, arch_config, mean_accuracy))
            if mean_accuracy > best_performance:
                best_performance = mean_accuracy
                best_params = (param_dict, arch_config)

        print(
            f"Best Parameters: {best_params[0]} with Architecture: {best_params[1]} and Mean Accuracy: {best_performance:.4f}")
        return results, best_params


    param_grid = {
        'lr': [0.001, 0.01, 0.0001, 0.1],  # Learning rate options
        'epochs': [5, 10, 15],  # Number of epochs to try
        'batch_size': [8, 16, 32, 64],  # Batch sizes to test
        'optimizer': [torch.optim.SGD, torch.optim.Adam],  # Optimizer options
        'pooling_after_conv': [True, False],  # Whether to pool after convolution
        'dropout_rate': [0.1, 0.3, 0.5, 0.7],  # Dropout rates
        'weight_decay': [0.0, 0.01, 1e-4, 1e-5]  # Weight decay (L2 regularization)
    }


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
    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        losses, accs, fbetas = [], [], []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = F.binary_cross_entropy(outputs, labels)
            acc = accuracy(outputs, labels)
            fbeta = fbeta_score(outputs, labels, beta=0.5)
            losses.append(loss.item())
            accs.append(acc.item())
            fbetas.append(fbeta)

        return {
            'loss': sum(losses) / len(losses),
            'accuracy': sum(accs) / len(accs),
            'f0.5': sum(fbetas) / len(fbetas)
        }


    def evaluate_final_model(train_data, test_data, best_params):
        # Unpack best hyperparameters and architecture
        param_dict, best_architecture = best_params
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']
        lr = param_dict['lr']
        optimizer = param_dict['optimizer']
        weight_decay = param_dict['weight_decay']

        # Create final DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, num_workers=0)

        # Train the final model using the best architecture
        model = ConfigurableCNN(best_architecture).to(device)
        fit(
            epochs=epochs,
            lr=lr,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            opt_func=optimizer
        )

        # Evaluate the trained model on the test set
        test_results = evaluate(model, test_loader)
        print("Final Test Results:", test_results)
        return test_results


    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform random search with cross-validation to find the best hyperparameters
    results, best_params = random_search_with_cv(
        train_data=train_data,
        param_grid=param_grid,
        architecture_space=architecture_space,
        n_iter=1,
        k=3
    )

    print(f"Best Hyperparameters: {best_params[0]}")
    print(f"Best Architecture: {best_params[1]}")

    # Train the final model using the best hyperparameters and architecture
    param_dict, best_architecture = best_params  # Unpack best hyperparameters and architecture

    # Extract hyperparameters from best_params
    batch_size = param_dict['batch_size']
    epochs = param_dict['epochs']
    lr = param_dict['lr']
    optimizer = param_dict['optimizer']

    # Create DataLoaders with the best batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # Initialize the final model with the best architecture
    model = ConfigurableCNN(arch_config=best_architecture).to(device)

    # Train the model with the best hyperparameters
    history = fit(
        epochs=epochs,
        lr=lr,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        opt_func=optimizer
    )

    # Evaluate the final model on the test set
    test_results = evaluate(model, test_loader)
    print("Final Test Results:", test_results)
