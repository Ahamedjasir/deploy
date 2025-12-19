import streamlit as st
from groq import Groq
from PIL import Image
import requests
import io
import PyPDF2
import docx

# ---------------- CONFIG ----------------
GROQ_API_KEY = ""
HF_API_KEY = ""

HF_URL = f"https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

groq_client = Groq(api_key=GROQ_API_KEY)

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

st.set_page_config(page_title="AI Chat", page_icon="🤖")
st.title("Chat with AI")

SYSTEM_PROMPT = "You are a helpful AI assistant."

# ---------------- SESSION INIT ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

if "last_image" not in st.session_state:
    st.session_state.last_image = None

# ---------------- MODE SELECT ----------------
mode = st.selectbox(
    "Mode",
    ["Chat", "Image Generation"],
    index=0
)




# ---------------- CHAT HISTORY (CHAT MODE ONLY) ----------------
if mode == "Chat":
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])

# ---------------- FILE UPLOAD (CHAT MODE ONLY) ----------------
file_text = ""
if mode == "Chat":
    uploaded_file = st.file_uploader(
        "📎 Attach file",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            file_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif uploaded_file.type == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            doc = docx.Document(uploaded_file)
            file_text = "\n".join(p.text for p in doc.paragraphs)

        elif uploaded_file.type == "text/plain":
            file_text = uploaded_file.read().decode("utf-8")

# ---------------- CHATGPT-STYLE INPUT ----------------
user_input = st.chat_input(
    "Type your message..." if mode == "Chat" else "Describe the image..."
)

if user_input:
    # ---------- IMAGE GENERATION ----------
    if mode == "Image Generation":
        with st.spinner("Generating image..."):
            response = requests.post(
                HF_URL,
                headers=headers,
                json={
                    "inputs": user_input,
                    "options": {"wait_for_model": True}
                },
                timeout=120
            )

        content_type = response.headers.get("content-type", "")

        if "image" in content_type:
            st.session_state.last_image = response.content
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption=user_input)
        else:
            st.error("Image generation failed")
            st.code(response.text)

    # ---------- CHAT ----------
    else:
        full_prompt = user_input
        if file_text:
            full_prompt += f"\n\nAttached file content:\n{file_text}"

        st.session_state.messages.append(
            {"role": "user", "content": full_prompt}
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state.messages,
            temperature=0.7,
            max_completion_tokens=2048,
            top_p=1
        )

        ai_reply = response.choices[0].message.content

        st.session_state.messages.append(
            {"role": "assistant", "content": ai_reply}
        )
        st.rerun()
if (
        mode == "Chat"and len(st.session_state.messages) > 1 and st.button("🔄 Reset Chat")):
            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            st.rerun()

# ---------------- DOWNLOAD IMAGE (ONLY AFTER GENERATION) ----------------
if mode == "Image Generation" and st.session_state.last_image:
    st.download_button(
        "⬇ Download Image",
        data=st.session_state.last_image,
        file_name="generated_image.png",
        mime="image/png"
    )
