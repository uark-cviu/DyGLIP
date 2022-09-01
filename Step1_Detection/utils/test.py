import os.path as osp
import os
from .log import get_logger
from .run import main, parse_args
import shutil

cur_path = os.getcwd()
logger = get_logger(__name__)


def test(args):
    logger.info('Testing with args: %s', args)
    return main(parse_args(args))


if __name__ == "__main__":
    videos = [("CAMPUS", "Auditorium"), ("CAMPUS", "Garden1"), ("CAMPUS", "Garden2"), ("CAMPUS", "Parkinglot"),
    ("EPFL", "Basketball"), ("EPFL", "Campus"), ("EPFL", "Laboratory"), ("EPFL", "Passageway"), ("EPFL", "Terrace"), (".", "PETS09"),
    ("MCT", "Dataset1"), ("MCT", "Dataset2"), ("MCT", "Dataset3"), ("MCT", "Dataset4")]
    videos = [("MCT", "Dataset1"), ("MCT", "Dataset2"), ("MCT", "Dataset3"), ("MCT", "Dataset4")]
    for a, b in videos:
        DATA_PATH = os.path.join("/data/", a, b)
        OUT_PATH = os.path.join(cur_path, "MCMT", "exp_bigger", a, b)
        if not osp.exists(OUT_PATH):
            # shutil.rmtree(OUT_PATH)
            os.makedirs(OUT_PATH)

        if not osp.exists(DATA_PATH):
            raise RuntimeError("cant find the aicity dataset")
        ARGS = [DATA_PATH, osp.join(OUT_PATH, 'bboxes.pkl')]
        test(ARGS)
