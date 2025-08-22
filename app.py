import streamlit as st
from record_audio import record_and_transcribe
import time

st.set_page_config(page_title="Compass - AI Health Assistant", page_icon="🧭", layout="centered")

st.title("Compass 🧭")
st.subheader("Your AI Health Assistant")
st.write("Describe your symptoms using text or your microphone to get a preliminary suggestion. This is a demo and not real medical advice.")

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

mode = st.radio("Choose input method:", ["Text", "Voice"], horizontal=True)

if mode == "Text":
    if prompt := st.chat_input("Type your symptoms here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Compass is thinking..."):
                ai_response = get_llm_advice(prompt)
                message_placeholder.markdown(ai_response)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

elif mode == "Voice":
    if st.button("🎙️ Click to Record and Ask"):
        with st.spinner("Recording for 5 seconds... Please speak now."):
            transcribed_text = record_and_transcribe(duration=5)
        
        st.success("Recording complete!")
        
        st.session_state.messages.append({"role": "user", "content": f"(From voice) {transcribed_text}"})
        with st.chat_message("user"):
            st.markdown(f"(From voice) {transcribed_text}")

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Compass is thinking..."):
                ai_response = get_llm_advice(transcribed_text)
                message_placeholder.markdown(ai_response)

        st.session_state.messages.append({"role": "assistant", "content": ai_response})
