# ממשק Streamlit: צ'אט על אתר "דירה לדירה".
# הרצה: streamlit run app.py

import truststore

# בדיקת תעודות SSL דרך Windows (נדרש בגלל סינון NetFree)
truststore.inject_into_ssl()

import chromadb
import requests
import streamlit as st

import config
import ingest
import rag


def login(email, password):
    """התחברות דרך השרת של האתר.
    מחזיר (משתמש, None) בהצלחה, או (None, הודעת שגיאה) בכישלון."""
    try:
        response = requests.post(
            config.LOGIN_URL,
            json={"email": email, "password": password},
            timeout=config.LOGIN_TIMEOUT,
        )
    except requests.RequestException:
        # השרת לא זמין או לא ענה בזמן
        return None, config.SERVER_UNAVAILABLE_MESSAGE

    if response.status_code == 200:
        # שומרים רק תפקיד ושם - לא טוקן, טלפון או מייל
        user = response.json()["user"]
        return {"role": user["role"], "full_name": user["fullName"]}, None
    if response.status_code in (400, 401):
        # הודעת השגיאה של השרת, בעברית
        try:
            return None, response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            return None, config.SERVER_UNAVAILABLE_MESSAGE
    return None, config.SERVER_UNAVAILABLE_MESSAGE


st.set_page_config(page_title="דירה לדירה - עוזר", page_icon="🏠")

# עברית מימין לשמאל
st.markdown(
    """
    <style>
    .stApp, .stSidebar, .stChatMessage, .stChatInput, .stExpander {
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

# המשתמש המחובר: None - אורח
if "user" not in st.session_state:
    st.session_state.user = None

# טופס התחברות בסרגל הצד, רק כשההתחברות פעילה ואין משתמש מחובר
if config.AUTH_ENABLED and st.session_state.user is None:
    with st.sidebar:
        st.subheader("התחברות")
        with st.form("login_form"):
            email = st.text_input("מייל")
            password = st.text_input("סיסמה", type="password")
            submitted = st.form_submit_button("התחברות")
        if submitted:
            with st.spinner("מתחבר... (אם השרת לא היה פעיל, זה עשוי לקחת עד דקה)"):
                user, error = login(email, password)
            if user:
                st.session_state.user = user
                # בכל התחברות - ההיסטוריה נמחקת
                st.session_state.messages = []
                st.rerun()
            else:
                st.error(error)

# משתמש מחובר: ברכה וכפתור התנתקות
if st.session_state.user is not None:
    with st.sidebar:
        st.subheader(f"שלום, {st.session_state.user['full_name']}")
        if st.button("התנתקות"):
            st.session_state.user = None
            # בכל התנתקות - ההיסטוריה נמחקת
            st.session_state.messages = []
            st.rerun()


def show_sources(sources):
    """אזור מקופל "מקורות": שם המסמך וקטע קצר מכל מקטע שנשלף."""
    with st.expander("מקורות"):
        for source in sources:
            st.markdown(f"**{source['title']}**")
            st.caption(source["snippet"] + "...")


def get_history():
    """זוגות (שאלה, תשובה) מההיסטוריה שעל המסך, כהקשר לשאלות המשך."""
    messages = st.session_state.messages
    return [
        (prev["content"], curr["content"])
        for prev, curr in zip(messages, messages[1:])
        if prev["role"] == "user" and curr["role"] == "assistant"
    ]


# היסטוריית הצ'אט. נפרדת לכל לשונית
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

# שאלה חדשה: נשלחת עם השאלות והתשובות הקודמות כהקשר
if question := st.chat_input("כתבו שאלה על האתר..."):
    # ההיסטוריה נלקחת לפני הוספת השאלה החדשה
    history = get_history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש תשובה..."):
            # התפקיד קובע אילו מסמכים ייכללו בחיפוש. בלי התחברות - אורח
            user = st.session_state.user
            role = user["role"] if user else "GUEST"
            response, sources = rag.answer(question, role, history)
        st.markdown(response)
        if sources:
            show_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
