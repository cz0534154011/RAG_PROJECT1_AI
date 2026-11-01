# בניית האינדקס: טעינת המסמכים, חלוקה למקטעים בשתי שיטות ושמירה ב-Chroma.
# הרצה: py ingest.py

import json
import re
from pathlib import Path

import truststore

# בדיקת תעודות SSL דרך Windows (נדרש בגלל סינון NetFree)
truststore.inject_into_ssl()

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def load_documents():
    """טעינת קבצי הטקסט, ולכל קובץ ה-metadata שלו מ-metadata.json."""
    with open(config.METADATA_FILE, encoding="utf-8") as f:
        all_metadata = json.load(f)

    def file_metadata(file_path):
        # התאמה לפי שם הקובץ; קובץ שחסר ב-metadata.json יעצור את הטעינה
        file_name = Path(file_path).name
        return {"file_name": file_name, **all_metadata[file_name]}

    reader = SimpleDirectoryReader(
        input_dir=config.DATA_DIR,
        required_exts=[".txt"],
        file_metadata=file_metadata,
        encoding="utf-8",
    )
    return reader.load_data()


def split_by_size(documents):
    """שיטה א: חלוקה לפי גודל. ה-metadata של המסמך עובר לכל מקטע."""
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.get_nodes_from_documents(documents)


def split_by_headings(documents):
    """שיטה ב: כל קטע (כותרת + תוכן) הוא מקטע אחד."""
    nodes = []
    for doc in documents:
        # השורה הראשונה היא כותרת המסמך ולא קטע
        body = doc.text.strip().split("\n", 1)[1]
        # קטעים מופרדים בשורה ריקה
        for section in re.split(r"\n\s*\n", body.strip()):
            heading = section.split("\n", 1)[0].strip()
            nodes.append(
                TextNode(text=section.strip(), metadata={**doc.metadata, "section": heading})
            )
    return nodes


def get_embed_model():
    """מודל ה-Embeddings - אותו מודל בבנייה ובחיפוש."""
    return GoogleGenAIEmbedding(model_name=config.EMBED_MODEL, api_key=config.GOOGLE_API_KEY)


def build_collection(nodes, collection_name):
    """מחיקת האוסף הקיים (אם יש), ויצירת וקטורים ושמירה מחדש ב-Chroma."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=get_embed_model())
    return collection.count()


# שם השיטה (כמו ב-config.py) -> פונקציית החלוקה שלה
SPLITTERS = {
    "size": split_by_size,
    "headings": split_by_headings,
}


if __name__ == "__main__":
    # בנייה מחדש של שני האוספים
    documents = load_documents()
    print(f"נטענו {len(documents)} מסמכים")
    for method, collection_name in config.COLLECTIONS.items():
        nodes = SPLITTERS[method](documents)
        saved = build_collection(nodes, collection_name)
        print(f"שיטה {method}: {len(nodes)} מקטעים, נשמרו {saved} באוסף {collection_name}")
