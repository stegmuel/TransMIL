from datasets.bracs_dataset import BracsTilesDataset
from torch.utils.data import DataLoader
from models.TransMIL import TransMIL
from sklearn.metrics import f1_score
import numpy as np
import argparse
import logging
import random
import wandb
import torch
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Train', add_help=False)
    parser.add_argument('--train_queries', default=['/media/thomas/Samsung_T5/BRACS/BRACS_bags/train/*/*.npy'],
                        type=str, help='Please specify path to the training data.')
    parser.add_argument('--val_queries', default=['/media/thomas/Samsung_T5/BRACS/BRACS_bags/val/*/*.npy'],
                        type=str, help='Please specify path to the validation data.')
    parser.add_argument('--test_queries', default=['/media/thomas/Samsung_T5/BRACS/BRACS_bags/test/*/*.npy'],
                        type=str, help='Please specify path to the validation data.')
    parser.add_argument('--output_dir', default='output', type=str, help='Please specify path to the output dir.')
    parser.add_argument('--checkpoint_name', default='checkpoint_40_0', type=str, help='Please specify path to the output dir.')
    parser.add_argument('--logger', default='logs/log.txt', type=str, help='Please specify path to the logs dir.')
    parser.add_argument('--batch_size', default=1, type=str, help='Please specify the patch size.')
    parser.add_argument('--n_classes', default=7, type=str, help='Please specify the number of classes.')
    parser.add_argument('--epochs', default=10, type=str, help='Please specify the number of epochs.')
    parser.add_argument('--seed', default=0, type=str, help='Please specify the seed.')
    return parser


def get_logger(logfile):
    """
    Create and return a logger to store ans display the training statuses.
    :param logfile: location where to write the log outputs.
    :return: new logger.
    """
    # Instantiate a logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train_mil(args):
    # Set the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Login to wandb
    wandb.init(project='trans-mil', entity='stegmuel')
    wandb.config.update(args)
    args = wandb.config

    # Get the datasets
    def my_collate(batch):
        samples, labels = list(*batch)
        return samples[None, :], labels

    # Get the datasets and loader
    train_dataset = BracsTilesDataset(args.train_queries)
    train_loader = DataLoader(train_dataset, collate_fn=my_collate, shuffle=True)
    val_dataset = BracsTilesDataset(args.val_queries)
    val_loader = DataLoader(val_dataset, collate_fn=my_collate)
    test_dataset = BracsTilesDataset(args.test_queries)
    test_loader = DataLoader(test_dataset, collate_fn=my_collate)

    # Get the model
    model = TransMIL(args.n_classes).cuda()

    # Get the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = None

    # Set the loss
    criterion = torch.nn.CrossEntropyLoss()

    # Loss function and stored losses
    best_val_f1 = 0.
    losses = {'train': [], 'valid': []}
    accuracies = {'train': [], 'valid': []}

    # Get the logger
    logger = get_logger(args.logger)

    # Train
    logger.debug('About to train for {} epochs.'.format(args.epochs))

    # Infer the end epoch of the current session
    for epoch in range(args.epochs):
        # Set the models in train mode
        model.train()

        # Initialize metrics, etc.
        batch_losses, batch_accuracies = [], []
        total_samples, total_corrects = 0, 0
        for j, batch in enumerate(train_loader):
            # Execute a training step
            loss, n_samples, n_corrects = train_step(model, optimizer, batch, criterion)

            # Compute the accuracy
            total_samples += n_samples
            total_corrects += n_corrects
            batch_accuracies.append(n_corrects.item() / n_samples)

            # Store the batch losses
            batch_losses.append(loss)
            batch_losses.append(loss)

            # Display status
            display_rate = 100
            if j % display_rate == 0:
                n_batches = len(train_loader.dataset) // train_loader.batch_size
                message = 'epoch: {}/{}, batch {}/{}: \n' \
                          '\t loss: {:.2e}, acc: {:.2e}, lr: {:.2e}' \
                    .format(epoch, args.epochs, j, n_batches,
                            np.mean(batch_losses[-display_rate:]),
                            np.mean(batch_accuracies),
                            optimizer.param_groups[0]["lr"])
                logger.debug(message)

        # Update the scheduler
        if scheduler is not None:
            scheduler.step()

        # Store the epoch loss
        losses['train'].append(np.mean(batch_losses))

        # Compute the current epoch's accuracy
        accuracies['train'].append(total_corrects / total_samples)

        # Wandb logging
        wandb.log({"train loss": losses['train'][-1], 'train acc': accuracies['train']})

        # Evaluate the model
        with torch.no_grad():
            val_f1 = test_mil(model, val_loader, logger, criterion, flag='val')
            save(model, args, val_f1 > best_val_f1)

    # Re-load the best model
    logger.debug('Re-loading best weights for the evaluation on the test set.'.format(args.epochs))
    model = load(model, args)

    # Test
    logger.debug('About to test the trained model.'.format(args.epochs))
    with torch.no_grad():
        test_mil(model, test_loader, logger, criterion, flag='val')


def save(model, args, is_best):
    # Prepare the dictionary
    save_dict = {
        'model_state_dict': model.state_dict()
    }

    # The model is overwritten at the end of every epoch
    savepath = os.path.join(args.output_dir, f"{args.checkpoint_name}.pth")
    torch.save(save_dict, savepath)

    # Save the best model
    if is_best:
        savepath = os.path.join(args.output_dir, f"{args.checkpoint_name}_best.pth")
        torch.save(save_dict, savepath)


def load(model, args):
    # Load the best state dict
    loadpath = os.path.join(args.output_dir, f"{args.checkpoint_name}_best.pth")
    checkpoint = torch.load(loadpath, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def train_step(model, optimizer, batch, criterion):
    """
    Executes a single step of training of the PAWS framework on the provided batch.
    :param batch: single batch of data.
    :return: losses and accuracy.
    """
    # Move data to device
    samples, labels = batch
    samples = samples.cuda()
    labels = labels.cuda()

    # Get patch level predictions
    outputs = model(samples)

    _, predicted = torch.max(outputs.data, 1)

    # Compute the loss as a convex combination
    loss = criterion(outputs, labels.unsqueeze(0))

    # Compute the #correct predictions
    n_corrects = predicted.eq(labels.data).cpu().sum().float()

    # Get the number of processed samples
    n_samples = outputs.shape[0]

    # Back-propagate the loss
    optimizer.zero_grad()
    loss.backward()

    # Update the model's weights
    optimizer.step()
    return loss.item(), n_samples, n_corrects


def test_mil(model, loader, logger, criterion, flag='val'):
    """
    Evaluates the performance of the model on the validation set.
    :param: result_savepath: location where to save model's predictions.
    :return: None.
    """
    # Set both models on evaluation mode
    model.eval()

    # Display start message
    logger.debug(f"Starting the evaluation on the {flag} set.")

    # Iterate over the batches in the validation set
    total_corrects, total_samples, predictions, targets, losses = 0, 0, [], [], []
    for batch in loader:
        # Move data to device
        samples, labels = batch
        samples = samples.cuda()
        labels = labels.cuda().unsqueeze(0)

        # Store the targets
        targets.append(labels.cpu())

        # Get the predictions
        outputs = model(samples)
        predicted = outputs.argmax(dim=-1)
        losses.append(criterion(outputs, labels))

        # Store the predictions
        predictions.append(torch.argmax(outputs, dim=-1).cpu())

        # Compute the accuracy
        total_corrects += predicted.eq(labels.data).cpu().sum().float()
        total_samples += outputs.shape[0]

    # Compute the F1-scores
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    weighted_f1 = f1_score(targets, predictions, average='weighted')
    class_f1 = f1_score(targets, predictions, average=None)

    # Wandb logging
    wandb_dict = {
        f"{flag} loss": losses[-1],
        f"{flag} f1": weighted_f1,
    }

    # Add the class F1-scores
    for k, v in zip(loader.dataset.class_dict.keys(), class_f1):
        wandb_dict[k] = v
        message = f"{flag}: f1-score for class {k}: {v}"
        logger.debug(message)
    wandb.log(wandb_dict)

    # Display the F1-score
    message = f"{flag}: f1-score: {weighted_f1}"
    logger.debug(message)
    return weighted_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data', parents=[get_args_parser()])
    args = parser.parse_args()
    train_mil(args)
