import torchvision.transforms as pth_transforms
from torchvision.models import resnet50
from einops import rearrange
from pathlib import Path
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Data', add_help=False)
    parser.add_argument('--image_queries', default=['/media/thomas/Samsung_T5/BRACS/BRACS_RoI/previous/*/*/*.png'],
                        type=str, help='Please specify path to the training data.')
    parser.add_argument('--output_dir', default='/media/thomas/Samsung_T5/BRACS/BRACS_bags', type=str,
                        help='Please specify path to the output dir.')
    parser.add_argument('--patch_size', default=256, type=str, help='Please specify the patch size.')
    parser.add_argument('--batch_size', default=64, type=str, help='Please specify the patch size.')
    return parser


def get_bags(args):
    # Get all the image paths
    image_paths = [f for q in args.image_queries for f in glob(q)]

    # Instantiate the model
    model = resnet50(pretrained=True).cuda()
    model.fc = torch.nn.Identity()
    model.eval()

    # Set the transforms
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Process all images
    for image_path in tqdm(image_paths):
        # Load the image
        image = Image.open(image_path)

        # The image size must be a multiple of the patch size
        image = transform(image)
        c, h, w = image.shape
        h_padded, w_padded = -(-h // args.patch_size) * args.patch_size, -(-w // args.patch_size) * args.patch_size
        h_diff, w_diff = h_padded - h, w_padded - w
        image_padded = torch.zeros(c, h_padded, w_padded)
        image_padded[:, h_diff // 2: (h_diff // 2) + h, w_diff // 2: w_diff // 2 + w] = image

        # Patchify the image
        patches = rearrange(image_padded, 'c (m h) (n w) -> (m n) c h w', h=args.patch_size, w=args.patch_size)

        # Split in batches
        n_chunks = -(-patches.shape[0] // args.batch_size)
        batches = patches.chunk(n_chunks)

        # Iterate over the batches
        for i, batch in enumerate(batches):
            if i > 1:
                continue
            # Get the batch's embeddings
            with torch.no_grad():
                embeddings = model(batch.cuda())

            # Save the embeddings
            image_name = image_path.split('/')[-1].split('.')[0]
            intermediate_dir = '/'.join(image_path.split('/')[7: 9])
            intermediate_dir = os.path.join(args.output_dir, intermediate_dir)
            for j, embedding in enumerate(embeddings.unbind()):
                tile_index = i * args.batch_size + j
                embedding_name = f"{image_name}_{tile_index}.npy"

                # Create the embedding dir if needed
                p = Path(intermediate_dir)
                p.mkdir(exist_ok=True, parents=True)

                # Save the embedding
                np.save(os.path.join(intermediate_dir, embedding_name), embedding.cpu())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data', parents=[get_args_parser()])
    args = parser.parse_args()
    get_bags(args)
