import sys
import numpy as np
import cv2

# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb

from query_data import classify_img, get_most_similar_chunks, create_response
from utils import DB_PATH

client_db = chromadb.PersistentClient(path=DB_PATH)

# set title
st.title('RAG APPLICATION')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # Convert the file buffer stream directly to an OpenCV format in-memory array matrix
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if opencv_image is None:
        st.error("Error decoding image structure. Try another file format.")
    else:
        # Pass the matrix image directly down the pipeline
        clf = classify_img(client_db, opencv_image)
        clf_ = clf.replace('_', ' ').title()

        st.write(f'You are currently looking at the **{clf_}** !\n Is there anything you would like to know about it?')

        # Unique key added to prevent text input state conflicts in Streamlit loops
        user_question = st.text_input(f'Ask a question about **{clf_}**:', key="user_query_input")

        # write agent response
        if user_question and user_question.strip() != "":
            with st.spinner(text="In progress..."):
                chunks, metadata = get_most_similar_chunks(client_db, user_question, clf)
                response, sources = create_response(chunks, metadata, user_question)

                st.write(response)