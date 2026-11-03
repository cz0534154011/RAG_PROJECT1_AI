# אחזור, סינון הרשאות ויצירת תשובה.

import truststore

# בדיקת תעודות SSL דרך Windows (נדרש בגלל סינון NetFree)
truststore.inject_into_ssl()

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from ingest import get_embed_model

# הנחיות למודל. {history} - השיחה הקודמת, {context} - המקטעים שנשלפו, {question} - השאלה
PROMPT_TEMPLATE = """אתה עוזר של אתר "דירה לדירה" להחלפת דירות לשבתות ולחופשות.
תפקידך לענות על שאלות על האתר, רק על סמך קטעי המידע שלמטה.

הנחיות:
1. ענה בעברית, בשפה חופשית ויפה.
2. ענה בצורה מדויקת וממוקדת, רק על מה שנשאל.
3. השתמש רק במידע שבקטעים. אל תסיק מסקנות שלא כתובות בהם ואל תוסיף ידע כללי.
4. אל תחשוף פרטים ניהוליים שאינם מופיעים בקטעים.
5. אל תמסור מידע על אנשים אחרים.
6. אם השאלה לא קשורה לאתר, או שהתשובה לא נמצאת בקטעים, החזר בדיוק את ההודעה הבאה ושום דבר מלבדה:
{no_info}
7. השיחה הקודמת נועדה רק כדי להבין למה השאלה מתייחסת. את התשובה עצמה בסס רק על קטעי המידע.

המשתמש ששואל: {user_role}. כל קטעי המידע שלמטה מותרים לו.

שיחה קודמת:
{history}

קטעי המידע:
{context}

שאלה: {question}
תשובה:"""

# ניסוח מחדש של שאלת המשך כשאלה עצמאית, לפני החיפוש
REWRITE_TEMPLATE = """לפניך שיחה קודמת ושאלה חדשה על אתר "דירה לדירה".
נסח את השאלה החדשה מחדש כשאלה עצמאית ושלמה בעברית, שאפשר להבין בלי השיחה הקודמת.
אם השאלה החדשה כבר מובנת לבד, או שהיא עוסקת בנושא אחר מהשיחה הקודמת, החזר אותה כמו שהיא.
החזר רק את השאלה, בלי הסברים.

שיחה קודמת:
{history}

שאלה חדשה: {question}
שאלה עצמאית:"""


def get_index(method=config.CHUNK_METHOD):
    """פתיחת האוסף של השיטה הפעילה ב-Chroma, עם אותו מודל Embeddings של הבנייה."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(config.COLLECTIONS[method])
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())


def retrieve(index, question, role):
    """שליפת TOP_K המקטעים הקרובים לשאלה, רק מתוך מה שהתפקיד רשאי לראות."""
    # הסינון נעשה בתוך Chroma, לפני שמקטע כלשהו נשלח למודל
    filters = MetadataFilters(filters=[
        MetadataFilter(key="audience", value=config.AUDIENCE_BY_ROLE[role], operator=FilterOperator.IN)
    ])
    retriever = index.as_retriever(similarity_top_k=config.TOP_K, filters=filters)
    return retriever.retrieve(question)


# אורך הקטע הקצר שמוצג מכל מקור
SNIPPET_LENGTH = 150


def format_history(history):
    """היסטוריה כטקסט: זוגות של (שאלה, תשובה), מהישן לחדש."""
    return "\n\n".join(f"שאלה: {q}\nתשובה: {a}" for q, a in history)


def rewrite_question(llm, question, history):
    """ניסוח שאלת המשך כשאלה עצמאית. בלי היסטוריה - השאלה חוזרת כמו שהיא."""
    if not history:
        return question
    prompt = REWRITE_TEMPLATE.format(history=format_history(history), question=question)
    return llm.complete(prompt).text.strip()


def answer(question, role, history=None, method=config.CHUNK_METHOD):
    """תשובה לשאלה ורשימת המקורות שנשלפו.
    history - רשימת זוגות (שאלה, תשובה) קודמים; נלקחים רק HISTORY_TURNS האחרונים."""
    history = (history or [])[-config.HISTORY_TURNS:] if config.HISTORY_TURNS else []
    llm = GoogleGenAI(model=config.LLM_MODEL, api_key=config.GOOGLE_API_KEY)

    # החיפוש והתשובה מקבלים את השאלה המנוסחת, כדי ששאלת המשך תהיה ברורה לשניהם
    search_question = rewrite_question(llm, question, history)
    nodes = retrieve(get_index(method), search_question, role)
    if not nodes:
        return config.NO_INFO_MESSAGE, []

    # כל מקטע עם שם המסמך שלו, כדי שהמודל יבין את ההקשר
    context = "\n\n".join(f"[{n.metadata['title']}]\n{n.text}" for n in nodes)
    prompt = PROMPT_TEMPLATE.format(
        no_info=config.NO_INFO_MESSAGE,
        user_role=config.ROLE_LABELS[role],
        history=format_history(history) or "אין",
        context=context,
        question=search_question,
    )
    response = llm.complete(prompt).text.strip()

    sources = [
        {"title": n.metadata["title"], "snippet": n.text[:SNIPPET_LENGTH]}
        for n in nodes
    ]
    return response, sources
