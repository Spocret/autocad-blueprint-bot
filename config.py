import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "blueprint_bot.db"))
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", os.path.join(BASE_DIR, "outputs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_IMAGE_SIZE = (2048, 2048)
CONFIDENCE_THRESHOLD = 0.75
