# כל ההגדרות של הפרויקט במקום אחד.

import os
from pathlib import Path

from dotenv import load_dotenv

# --- נתיבים ---
# תיקיית הפרויקט: הנתיבים נכונים מכל תיקייה שממנה מריצים
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# --- מודלים ומפתח API ---
# המפתח נקרא מ-.env בלבד ולא נכתב בקוד
load_dotenv(BASE_DIR / ".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-2"

# --- חלוקה למקטעים ו-Chroma ---
# שיטה א - לפי גודל (SentenceSplitter)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# אוסף נפרד ב-Chroma לכל שיטה
COLLECTIONS = {
    "size": "chunks_size",          # שיטה א - לפי גודל
    "headings": "chunks_headings",  # שיטה ב - לפי כותרות
}

# השיטה שהאפליקציה משתמשת בה: "size" או "headings"
CHUNK_METHOD = "headings"

# --- אחזור והרשאות ---
# כמה מקטעים נשלפים לכל שאלה
TOP_K = 3

# ערכי audience שכל תפקיד רשאי לראות (סינון לפני השליחה למודל)
AUDIENCE_BY_ROLE = {
    "GUEST": ["all", "user"],          # אורח
    "USER": ["all", "user"],           # משתמש רשום
    "ADMIN": ["all", "user", "admin"], # מנהל
}

# --- זיכרון קצר ---
# כמה שאלות ותשובות אחרונות נשלחות למודל כהקשר
HISTORY_TURNS = 3

# --- תשובה ---
# ההודעה המדויקת כשאין מידע במאגר
NO_INFO_MESSAGE = "לא נמצא מידע על כך במאגר של האתר. ניתן לפנות למנהל האתר דרך דף 'צור קשר'."
# כש-Gemini לא זמין, עמוס או שהמכסה נגמרה
AI_UNAVAILABLE_MESSAGE = "השירות עמוס כרגע או שהגיע למכסת השימוש. ניתן לנסות שוב בעוד כמה דקות."

# --- התחברות ---
# False: אין טופס התחברות וכולם אורחים
AUTH_ENABLED = True
LOGIN_URL = "https://apartment-swap-api.onrender.com/api/auth/login"
# זמן המתנה לשרת בשניות (השרת החינמי "נרדם" ועשוי לענות אחרי כדקה)
LOGIN_TIMEOUT = 60
SERVER_UNAVAILABLE_MESSAGE = "השרת של האתר אינו זמין כרגע. ניתן להמשיך כאורח."
