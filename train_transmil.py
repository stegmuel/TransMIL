from models.TransMIL import TransMIL
import numpy as np
import argparse
import logging
import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Train', add_help=False)
    parser.add_argument('--train_queries', default=['/media/thomas/Samsung_T5/BRACS/BRACS_bags/train/*/*.npy'],
                        type=str, help='Please specify path to the training data.')
    parser.add_argument('--val_queries', default=['/media/thomas/Samsung_T5/BRACS/BRACS_bags/val/*/*.npy'],
                        type=str, help='Please specify path to the validation data.')
    parser.add_argument('--test_queries', default=['/media/thomas/Samsung_T5/BRACS/BRACS_bags/test/*/*.npy'],
                        type=str, help='Please specify path to the validation data.')
    parser.add_argument('--output_dir', default='/media/thomas/Samsung_T5/BRACS/BRACS_bags', type=str,
                        help='Please specify path to the output dir.')
    parser.add_argument('--batch_size', default=1, type=str, help='Please specify the patch size.')
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
    # Get the datasets
    train_dataset =
    train_loader =
    val_dataset =
    val_loader =

    # Get the model
    model =

    # Get the optimizer
    optimizer =
    scheduler = None

    # Set the loss
    criterion =

    # Loss function and stored losses
    best_valid_f1_score = 0.
    losses = {'train': [], 'valid': []}
    batch_losses = {'train': [], 'valid': []}
    accuracies = {'train': [], 'valid': []}
    f1_scores = {'weighted': [], 'class': []}

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
            batch_losses['train'].append(loss)

            # Display status
            display_rate = 10
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

        # Evaluate the model
        with torch.no_grad():
            test_mil(model, val_loader)

        # Terminate epoch
        # self.on_epoch_end(end_epoch)

    # Save the final state
    # self.save(True)


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
    loss = criterion(outputs, labels)

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


def test_mil(args, model, loader):
    """
    Evaluates the performance of the model on the validation set.
    :param: result_savepath: location where to save model's predictions.
    :return: None.
    """
    # Set both models on evaluation mode
    model.eval()

    # Display start message
    logger.debug('Starting the evaluation on the test set.')

    # Iterate over the batches in the validation set
    total_corrects, total_samples, predictions, targets = 0, 0, [], []
    for batch in loader:
        # Move data to device
        samples, labels = batch
        samples = samples.cuda()
        labels = labels.cuda()

        # Store the targets
        targets.append(labels.cpu())

        # Get the predictions
        outputs = model(samples)

        # Store the predictions
        predictions.append(torch.argmax(outputs, dim=-1).cpu())

        # Compute the accuracy
        total_corrects += correct_predictions(outputs, labels)
        total_samples += outputs.shape[0]

    # Compute the F1-scores
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    weighted_test_f1_score = f1_score(targets, predictions, average='weighted')
    class_test_f1_score = f1_score(targets, predictions, average=None)

    # Display the F1-score
    message = 'test: f1-score: {:.2e}'.format(weighted_test_f1_score)
    logger.debug(message)

    # Add the class F1-scores
    for k, v in zip(loader.dataset.class_dict.keys(), class_test_f1_score):
        message = 'test: f1-score for class {}: {:.2e}'.format(k, v)
        logger.debug(message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data', parents=[get_args_parser()])
    args = parser.parse_args()
    train_mil(args)
