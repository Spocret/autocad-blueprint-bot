import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "blueprint_bot.db"))
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", os.path.join(BASE_DIR, "outputs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_IMAGE_SIZE = (2048, 2048)
CONFIDENCE_THRESHOLD = 0.75
