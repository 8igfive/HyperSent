import os
import logging
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, List
from geoopt import PoincareBallExact
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
Manifold = PoincareBallExact()

@dataclass
class VisualizeEmbeddingsArguments:
    visualize_embeddings: bool = field(
        default=False,
        metadata={'help': 'Whether to visualize embeddings.'}
    )
    embeddings_path: str = field(
        default=None,
        metadata={'help': 'Path to Hyperbolic Embeddings'}
    )
    dim_len: int = field(
        default=2,
        metadata={
            'help': 'Dimensional of visualization.',
            'choices': [2, 3]
        }
    )
    max_num: int = field(
        default=100, 
        metadata={'help': 'Maximum visualized embeddings number.'}
    )
    save_name: str = field(
        default=None,
        metadata={'help': 'Name of the saved figure.'}
    )

def visualize_embeddings(args: VisualizeEmbeddingsArguments):
    if not args.visualize_embeddings:
        return
    assert args.dim_len == 2 or args.dim_len == 3, f"Only support 2/3 dimensional visualization."
    assert os.path.isfile(args.embeddings_path), f"Need to be a path to embeddings file, but got {args.visualize_embeddings}"
    with open(args.embeddings_path, 'r', encoding='utf8') as fo:
        embeddings = json.load(fo)
        logger.info('Done reading.')
    
    embeddings = torch.tensor(embeddings[: args.max_num])
    if args.dim_len < embeddings.shape[-1]:    
        embeddings = Manifold.logmap0(embeddings, dim=-1)
        embeddings = torch.svd(embeddings)[0][:, :args.dim_len]
        embeddings = Manifold.expmap0(embeddings, dim=-1)
        logger.info('Done SVD.')

        embeddings = embeddings / (embeddings.norm(dim=-1, p=2).max() * 1.01)

    if args.dim_len == 2:
        fig, ax = plt.subplots(figsize=(16, 16))
        theta = np.linspace(0, 2 * np.pi, 200)
        x = 1 * np.cos(theta)
        y = 1 * np.sin(theta)
        ax.plot(x, y, color="black", linewidth=1)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color='darkblue', linewidths=0.5)
        ax.xaxis.set_major_locator(plt.NullLocator()) 
        ax.yaxis.set_major_locator(plt.NullLocator()) 
    else:
        fig = plt.figure(figsize=(16, 16))
        ax = Axes3D(fig)
        ax.scatter3D(embeddings[:, 0], embeddings[:, 1], embeddings[:, 1], color='darkblue', linewidths=0.5)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.zaxis.set_major_locator(plt.NullLocator())
    save_name = args.save_name
    if not save_name:
        save_name = f'{args.dim_len}-dimensional.pdf'
    save_dir = os.path.join('results', 'visualize_embeddings')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_name))
    logger.info(f'Done saving, save_path={os.path.join(save_dir, save_name)}')