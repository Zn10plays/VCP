from architecture import pipe
from dataloader.original import get_dataset
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == '__main__':
    logger = TensorBoardLogger('logs', name='LViTLogs')
    trainer = Trainer(logger=logger, max_epochs=100)

    training_data = DataLoader(get_dataset('train', augmentation=True),
                               batch_size=32,
                               shuffle=True,
                               num_workers=8,
                               persistent_workers=True)

    testing_data = DataLoader(get_dataset('test', augmentation=True),
                              batch_size=32,
                              num_workers=8,
                              persistent_workers=True)

    model = pipe.get_model()

    trainer.fit(model, training_data, testing_data)