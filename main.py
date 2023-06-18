import streamlit as st
import json
import numpy as np
from tensorflow import keras
import pickle

import colorama
colorama.init()
from colorama import Fore, Style

with open("C:\\Users\\ASUS\\Desktop\\Deeplearning chatbot\\ProjekAkhirDeepLearning\\intent.json") as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model('chat_model')

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20


def get_chat_response(user_input):
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return ""


def main():
    st.title("Chatbot reservasi hotel")

    st.markdown("<h3 style='color: yellow; text-align: center;'>Start messaging with the bot!</h3>",
                unsafe_allow_html=True)

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = st.text_input("You:", "", key="input")

    if user_input:
        if user_input.lower() == "quit":
            st.stop()

        output = get_chat_response(user_input)

        # Store the output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        user_input = ""

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            col_chatbot, col_user = st.columns(2)

            with col_chatbot:
                st.markdown("<div style='display: flex; justify-content: flex-start;'>"
                            "<div style='background-color: #DCF8C6; color: black; border-radius: 5px; "
                            "padding: 8px; margin-bottom: 5px;'>{}</div></div>".format(
                                st.session_state['generated'][i]), unsafe_allow_html=True)

            with col_user:
                st.markdown("<div style='display: flex; justify-content: flex-end;'>"
                            "<div style='background-color: #E8F0FE; color: black; border-radius: 5px; "
                            "padding: 8px; margin-bottom: 5px;'>{}</div></div>".format(
                                st.session_state['past'][i]), unsafe_allow_html=True)


if __name__ == '__main__':
    main()