# Imports from external libraries

# Imports from internal libraries

DATA_ROOT = "/run/media/erik/A0BE428BBE425A44/DataWearhouseFast/VoxCeleb/"
TRAIN_DATA = f"{DATA_ROOT}" + "dev/dev/aac"
TEST_DATA = f"{DATA_ROOT}" + "test/aac"
DATABASE = f"{DATA_ROOT}" + "test.db"
IMAGES = "images/"

PRETRAINED_MODE = "test"

INT_16_MAX  = (2 ** 15) - 1

SR = 16000
TAR_DBFS = -30

VAD_WIN_LEN = 30
VAD_AVERAGE = 8
VAD_MAX_SILENCE  = 6

MEL_WIN_LEN = 25
MEL_WIN_STEP = 10
MEL_CHANNELS = 40

PART_UTTER = 160