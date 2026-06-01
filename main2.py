import logging
import os
import multiprocessing as mp
# import synpp
import sys

sys.path.append("/cluster/project/cmdp/dabdelkader/cmdp_github/eqasim-switzerland/synpp/synpp")
from src.synpp.pipeline import Synpp


logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s %(levelname)s %(name)s: %(message)s",
       stream=sys.stdout,
       force=True,
   )

################ Run the pipeline ################

CONFIG_PATH = "/cluster/project/cmdp/dabdelkader/cmdp_github/eqasim-switzerland/config_dib_2.yml"

if not os.path.isfile(CONFIG_PATH):
    raise FileNotFoundError(f"Config file does not exist: {CONFIG_PATH}")


def main():
    pipeline = Synpp.build_from_yml(config_path=CONFIG_PATH)
    pipeline.run_pipeline()

if __name__ == "__main__":
    mp.freeze_support()
    main()
