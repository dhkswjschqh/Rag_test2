import streamlit as st
import tiktoken

from langchain_core.messages import ChatMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langserve import RemoteRunnable


# ---------------------------------------------------------
# 토큰 길이 계산 함수
# ---------------------------------------------------------
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


# ---------------------------------------------------------
# 업로드된 PDF 파일 읽기
# ---------------------------------------------------------
def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name

        # 업로드된 파일을 임시로 저장
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())

        if ".pdf" in doc.name.lower():
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
            doc_list.extend(documents)

    return doc_list


# ---------------------------------------------------------
# 문서를 chunk 단위로 분할
# ---------------------------------------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )

    chunks = text_splitter.split_documents(text)

    return chunks


# ---------------------------------------------------------
# FAISS Vector Store 생성
# ---------------------------------------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = FAISS.from_documents(
        text_chunks,
        embeddings
    )

    return vectordb


# ---------------------------------------------------------
# 검색된 문서를 하나의 context 문자열로 합치기
# ---------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    st.set_page_config(
        page_title="StreamLit_remote_RAG",
        page_icon="📚"
    )

    st.title("📚 RAG_test4 Q/A Chat")


    # -----------------------------------------------------
    # Session State 초기화
    # -----------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "store" not in st.session_state:
        st.session_state["store"] = dict()

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    if "retriever" not in st.session_state:
        st.session_state.retriever = None


    # -----------------------------------------------------
    # 채팅 기록 출력
    # -----------------------------------------------------
    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)


    # -----------------------------------------------------
    # 채팅 기록 추가
    # -----------------------------------------------------
    def add_history(role, content):
        st.session_state.messages.append(
            ChatMessage(
                role=role,
                content=content
            )
        )


    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------
    with st.sidebar:

        uploaded_files = st.file_uploader(
            "Upload your file",
            type=["pdf"],
            accept_multiple_files=True
        )

        process = st.button("Process")


    # -----------------------------------------------------
    # PDF 처리
    # -----------------------------------------------------
    if process:

        if not uploaded_files:
            st.warning("PDF 파일을 먼저 업로드해주세요.")

        else:

            with st.spinner("문서를 처리하고 있습니다..."):

                files_text = get_text(uploaded_files)

                text_chunks = get_text_chunks(files_text)

                vectorstore = get_vectorstore(text_chunks)

                retriever = vectorstore.as_retriever(
                    search_type="mmr"
                )

                st.session_state["retriever"] = retriever

                st.session_state.processComplete = True

            st.success("문서 처리가 완료되었습니다.")


    # -----------------------------------------------------
    # 첫 안내 메시지
    # -----------------------------------------------------
    if len(st.session_state.messages) == 0:

        add_history(
            "assistant",
            "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
        )


    # -----------------------------------------------------
    # RAG Prompt
    # -----------------------------------------------------
    RAG_PROMPT_TEMPLATE = """
당신은 동서울대학교 컴퓨터소프트웨어학과 안내 AI입니다.

검색된 문맥을 참고하여 사용자의 질문에 답변하세요.

검색된 문맥에 답이 존재하면 그 내용을 바탕으로 정확하게 답변하세요.
검색된 문맥에서 답을 찾을 수 없다면 모른다고 답변하세요.

Question:
{question}

Context:
{context}

Answer:
"""


    # 기존 채팅 기록 출력
    print_history()


    # -----------------------------------------------------
    # 사용자 입력
    # -----------------------------------------------------
    if user_input := st.chat_input("메시지를 입력해 주세요"):

        # 사용자 메시지 저장
        add_history(
            "user",
            user_input
        )

        # 사용자 메시지 화면 표시
        st.chat_message("user").write(user_input)


        # -------------------------------------------------
        # AI 응답
        # -------------------------------------------------
        with st.chat_message("assistant"):

            # =================================================
            # 핵심
            #
            # Streamlit
            #    ↓
            # ngrok
            #    ↓
            # LangServe
            #    ↓
            # Ollama
            #    ↓
            # llama3-unslot-q8
            # =================================================
            llm = RemoteRunnable(
                "https://elated-distort-dyslexic.ngrok-free.dev/llm/"
            )

            chat_container = st.empty()


            # -------------------------------------------------
            # RAG가 활성화된 경우
            # -------------------------------------------------
            if st.session_state.processComplete is True:

                prompt1 = ChatPromptTemplate.from_template(
                    RAG_PROMPT_TEMPLATE
                )

                retriever = st.session_state["retriever"]


                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | prompt1
                    | llm
                    | StrOutputParser()
                )


                # =============================================
                # stream() 사용하지 않음
                # 현재 LangServe 버전에서 JSONDecodeError 방지
                # =============================================
                final_answer = rag_chain.invoke(
                    user_input
                )


                chat_container.markdown(
                    final_answer
                )


                add_history(
                    "assistant",
                    final_answer
                )


            # -------------------------------------------------
            # RAG가 비활성화된 경우
            # 맞춤형 LLM 자체에게 질문
            # -------------------------------------------------
            else:

                prompt2 = ChatPromptTemplate.from_template(
                    """
다음 질문에 한국어로 간결하게 답변해 주세요.

질문:
{input}

답변:
"""
                )


                chain = (
                    prompt2
                    | llm
                    | StrOutputParser()
                )


                # =============================================
                # stream() 대신 invoke()
                # =============================================
                final_answer = chain.invoke(
                    {
                        "input": user_input
                    }
                )


                chat_container.markdown(
                    final_answer
                )


                add_history(
                    "assistant",
                    final_answer
                )


# ---------------------------------------------------------
# 프로그램 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    main()