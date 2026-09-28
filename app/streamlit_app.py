import streamlit as st
from rag import AyurVanaRAG

st.set_page_config(
    layout="wide"
)

st.title("Aṣṭāṅga Saṃgraha RAG")
st.caption("Code-first, source-grounded prototype")

question = st.text_area(
    "Ask a question about the book",
    placeholder="e.g. Name different plants and how can they cure different diseases"
)

k = st.slider(
    "Number of retrieved passages",
    min_value=3,
    max_value=10,
    value=6
)

candidate_k = st.slider(
    "Candidate passages",
    min_value=10,
    max_value=30,
    value=18
)

if st.button("Ask") and question.strip():

    with st.spinner("Searching the book..."):

        # Create a fresh RAG object for each query during development.
        rag = AyurVanaRAG()

        answer, sources = rag.answer(
            question.strip(),
            k=k,
            candidate_k=candidate_k
        )

    st.markdown("## Answer")
    st.markdown(answer)

    st.markdown("---")

    with st.expander("Retrieved passages", expanded=True):

        if not sources:
            st.warning("No source passages were retrieved.")

        for i, (doc, meta, distance) in enumerate(sources, 1):

            page = meta.get("page", "Unknown")
            chunk = meta.get("chunk", "Unknown")

            st.markdown(
                f"### Source {i}"
            )

            st.write(
                f"**Page:** {page}  \n"
                f"**Chunk:** {chunk}  \n"
                f"**Distance:** {distance:.4f}"
            )

            st.write(doc)
            st.markdown("---")