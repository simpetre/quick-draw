import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import QuickDrawDataset
from model import ConvNet

class Trainer:
    """Trainer class for handling training and evaluation.
    
    Attributes:
        root (str): Path to the root directory containing the dataset.
        model (nn.Module): The neural network model to train.
        optimizer (Optimizer): The optimizer for training.
        criterion (nn.Module): The loss function.
        batch_size (int): Batch size for training.
    """
    
    def __init__(self, root: str, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, batch_size: int = 64):
        self.root = root
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.writer = SummaryWriter(log_dir='./logs')
        self._prepare_data()

    def _prepare_data(self):
        """Initialize the dataset and data loaders."""
        dataset = QuickDrawDataset(img_dir=self.root)
        train_size = int(0.01 * len(dataset))
        test_size = int(0.005 * len(dataset))
        remainder = len(dataset) - train_size - test_size

        self.train_dataset, self.test_dataset, _ = random_split(dataset, [train_size, test_size, remainder])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, num_epochs: int):
        """Train the model for a given number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to train for.
        """
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

            train_accuracy = 100. * correct_train / total_train
            print(f"End of epoch {epoch+1}, training accuracy: {train_accuracy:.2f}%")

    def evaluate(self):
        """Evaluate the model on the test dataset."""
        self.model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        test_accuracy = 100. * correct_test / total_test
        avg_test_loss = test_loss / len(self.test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    root = "/Users/simonpetre/datasets/quickdraw/"
    model = ConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(root=root, model=model, optimizer=optimizer, criterion=criterion)
    trainer.train(num_epochs=10)
    trainer.evaluate()
