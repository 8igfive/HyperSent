import logging
from transformers import HfArgumentParser
from hypersent.utils import visualize_embeddings, VisualizeEmbeddingsArguments

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((VisualizeEmbeddingsArguments,))
    ve_args: VisualizeEmbeddingsArguments = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        # "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    visualize_embeddings(ve_args)

if __name__ == '__main__':
    main()