import logging
from transformers import logging as transformers_logging
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
logger.setLevel(logging.INFO)
