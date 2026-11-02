# ממשק Streamlit: צ'אט על אתר "דירה לדירה".
# הרצה: streamlit run app.py

import chromadb
import streamlit as st

import config
import ingest
import rag

st.set_page_config(page_title="דירה לדירה - עוזר", page_icon="🏠")

# עברית מימין לשמאל
st.markdown(
    """
    <style>
    .stApp, .stChatMessage, .stChatInput, .stExpander {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠 דירה לדירה - עוזר")
st.caption("שאלות על האתר להחלפת דירות לשבתות ולחופשות")


@st.cache_resource(show_spinner="בונה את מאגר המידע, זה עשוי לקחת כמה שניות...")
def ensure_index():
    """אם האוסף הפעיל ריק או לא קיים - בנייה שלו לפני השימוש. רץ פעם אחת."""
    collection_name = config.COLLECTIONS[config.CHUNK_METHOD]
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if client.get_or_create_collection(collection_name).count() == 0:
        nodes = ingest.SPLITTERS[config.CHUNK_METHOD](ingest.load_documents())
        ingest.build_collection(nodes, collection_name)


ensure_index()


def show_sources(sources):
    """אזור מקופל "מקורות": שם המסמך וקטע קצר מכל מקטע שנשלף."""
    with st.expander("מקורות"):
        for source in sources:
            st.markdown(f"**{source['title']}**")
            st.caption(source["snippet"] + "...")


# היסטוריית הצ'אט - לתצוגה בלבד, לא נשלחת למודל. נפרדת לכל לשונית
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

# שאלה חדשה: נשלחת לבד, בלי השאלות הקודמות
if question := st.chat_input("כתבו שאלה על האתר..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש תשובה..."):
            # עד שלב ההתחברות - כולם אורחים
            response, sources = rag.answer(question, "GUEST")
        st.markdown(response)
        if sources:
            show_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
