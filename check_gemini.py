# בדיקה שהמפתח ומודלי Gemini עובדים.
# הרצה בלי פרמטרים: מציגה את המודלים הזמינים.
# הרצה עם שני פרמטרים: py check_gemini.py <מודל_שפה> <מודל_embeddings> - בודקת את שני המודלים.

import os
import sys

import truststore

# בדיקת תעודות SSL דרך Windows (נדרש בגלל סינון NetFree)
truststore.inject_into_ssl()

from dotenv import load_dotenv
from google import genai


def get_api_key():
    """קריאת המפתח מ-.env. עצירה עם הודעה ברורה אם חסר."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("שגיאה: GOOGLE_API_KEY לא נמצא בקובץ .env")
    return api_key


def list_models(api_key):
    """הצגת מודלי השפה ומודלי ה-Embeddings הזמינים למפתח."""
    client = genai.Client(api_key=api_key)
    llm_models = []
    embed_models = []
    for model in client.models.list():
        actions = model.supported_actions or []
        name = model.name.removeprefix("models/")
        if "generateContent" in actions:
            llm_models.append(name)
        if "embedContent" in actions:
            embed_models.append(name)

    print("המפתח תקין.\n")
    print("מודלי שפה (generateContent):")
    for name in llm_models:
        print("  ", name)
    print("\nמודלי Embeddings (embedContent):")
    for name in embed_models:
        print("  ", name)


def test_models(api_key, llm_name, embed_name):
    """שליחת בקשה קצרה לכל אחד מהמודלים שנבחרו, דרך LlamaIndex."""
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI

    llm = GoogleGenAI(model=llm_name, api_key=api_key)
    answer = llm.complete("ענה במילה אחת בעברית: מה צבע השמיים ביום בהיר?")
    print(f"מודל שפה {llm_name} ענה:", answer.text.strip())

    embed_model = GoogleGenAIEmbedding(model_name=embed_name, api_key=api_key)
    vector = embed_model.get_text_embedding("בדיקה")
    print(f"מודל Embeddings {embed_name} החזיר וקטור באורך {len(vector)}")


if __name__ == "__main__":
    key = get_api_key()
    if len(sys.argv) == 3:
        test_models(key, sys.argv[1], sys.argv[2])
    else:
        list_models(key)
