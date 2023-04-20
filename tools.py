import pdb
import logging
from transformers import HfArgumentParser
from hypersent.utils import visualize_embeddings, VisualizeEmbeddingsArguments, parse_log, ParseLogArguments, CheckEmbedAndCalSimArguments, check_embedding, cal_similarity

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((VisualizeEmbeddingsArguments, ParseLogArguments, CheckEmbedAndCalSimArguments))
    ve_args, pl_args, ce_cs_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        # "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    visualize_embeddings(ve_args)

    parse_log(pl_args)
    
    check_embedding(ce_cs_args)
    cal_similarity(ce_cs_args)

if __name__ == '__main__':
    main()
