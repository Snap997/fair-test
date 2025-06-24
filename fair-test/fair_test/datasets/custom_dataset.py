
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets import FederatedDataset

class Custom_Dataset():
    def __init__(self, id: str, name: str, n_classes: int, n_features: int):
        self.id = id
        self.name = name
        self.n_classes = n_classes  # Number of classes in the dataset
        self.n_features = n_features  # Number of features in dataset
        
    def load(self, num_partitions: int, alpha: float):
        
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=self.get_sensitive_feature(),
            alpha=alpha,
            seed=42,
            min_partition_size=0,
        )
        fds = FederatedDataset(
            dataset=self.id,
            partitioners={"train": partitioner},
        )
        return fds
    
    def get_n_classes(self):
        return self.n_classes
    
    def get_n_features(self):
        return self.n_features
    
    def get_sensitive_feature(self):
        """
        Returns the sensitive features of the dataset.
        This method should be overridden by subclasses to return the actual sensitive features.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    