import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH", "blueprint_bot.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
MAX_IMAGE_SIZE = (2048, 2048)
CONFIDENCE_THRESHOLD = 0.75
