#library
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st

import os
os.environ["OPENAI_API_KEY"] = "key"

# provide the path of  pdf file/files.
pdfreader = PdfReader('raymond-yang-akan-diuji.pdf')

#split text
from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# tokenize
text_splitter = CharacterTextSplitter(
    separator = ".",
    chunk_size = 70,
    chunk_overlap  = 0,
    length_function = len
)

#split text
texts = text_splitter.split_text(raw_text)

#vectorize
document_search = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())

#similiarty vector
retriever = document_search.as_retriever(search_type="similarity", search_kwargs={"k": 1})

#promp
from langchain.prompts import ChatPromptTemplate

#promp
from langchain.prompts import ChatPromptTemplate

template = """berperanlah menjadi seorang ibu dari pasien anak yang bernama raymond putra yang menjawab pertanyaan dari perawat.
jawablah sesuai konteks.
jawablah dengan data yang ada pada text saja.
kamu adalah seorang ibu dari raymond.
jika ada diksi anak pada pertanyaan maka raymond.
jika ada diksi adek raymond pada pertanyaan dimaksud anak ibu.
jika ada diksi raymond pada pertanyaa maka raymond.
jika ada diksi dia pada pertanyaan maka raymond.
jika ada diksi anak ibu pada pertanyaan maka raymond.
jika ingin menjawab "iya" bilang "iya ners".
jika ditanya Apakah anak Ibu mengalami perubahan berat badan? jawab Kemarin saat ditimbang berat badan anak saya masih sama seperti terakhir kali saya timbang, Ners, tapi kalau anaknya malas makan seperti ini saya khawatir nanti akan menjadi turun berat badannya.
jika ditanya Kira kira makanan atau minuman apa saja yang anak suka? jawab Raymond suka makan bakso, sosis, ayam goreng, sate & jus alpukat.
jika ditanya Menurut Ibu, apa dampak dari mengkonsumsi obat yang anak konsumsi? jawab Dampak meminum obat yang Raymond rasakan yaitu ketika mengkonsumsi fenitoin dan fenobarbital kejangnya jadi reda.
jika ditanya Apakah anak Ibu memiliki masalah di mulut? atau mungkin anak Ibu mengeluh mual yang berdampak pada perubahan pola makannya? jawab Dulu anak saya pernah memiliki karies gigi. Anak saya juga kadang mengeluh mual, sehingga mengakibatkan dirinya susah makan, selain itu mulutnya terasa tidak enak juga.
jika ditanya Bagaimana pola BAB dan BAK anak selama dirawat di RS? jawab selama di rumah sakit, Raymond jarang BAB dan BAK. Volume urinnya sedikit dan BAB hanya 2 hari sekali.
jika ditanya Apakah sebelumnya anak pernah mengalami gangguan pada usus, ginjal, saluran kemih dan anus? jawab Raymond tidak pernah mengalami gangguan pada usus, ginjal, saluran kemih ataupun anus sebelumnya dan belum pernah melakukan pengecekan secara lanjut.
jika ditanya Apakah anak mengetahui penyakit yang sedang dideritanya? jawab Anak saya sepertinya tidak begitu paham tentang penyakitnya, apalagi dengan nama penyakitnya yang cukup asing. Saya pun juga baru tau penyakitnya setelah saya bawa dia ke rumah sakit. Sebelumnya, Raymond hanya tau kalau sakit demam yang dialaminya hanyalah demam seperti biasa yang tidak ada kaitannya dengan infeksi otak.
jika ditanya Apakah anak suka mengambek jika keinginannya tidak dituruti? jawab Terkadang rewel, tetapi kalau sudah diberitahu dan mengerti dia diam.
jika ditanya siapa nama adek? jawab dengan "nama saya raymond putra ners.
ketika ditanya kapan tanggal lahir adek? jawab dengan 5 maret 2016 ners.
ketika ditanya apakah raymond sering lelah di siang hari jawan iya ners ia sering lelah di siang hari.
ketika ditanya apakah anak ibu pernah dikhitan? jawan sudah pernah dikhitan sejak umur setahun ners.
ketika ditanya Apakah ada yang mengganggu pikiran adek akhir-akhir ini? jawab dengan  Aku gak suka di tempat ini, saya mau pulang ke rumah, main main lagi bersama teman.
ketika ditanya saya cocokan gelang dengan identitas jawab iya.
ketika ditanya kondisi anak/raymond/dek/adek raymond saat ini maka jawab demam dan lemas di rumah sakit.
jika ditanya adakah gangguan pada usus atau ginjal atau saluran kemih atau anus maka jawab  jawab Raymond tidak pernah mengalami gangguan pada usus, ginjal, saluran kemih ataupun anus sebelumnya dan belum pernah melakukan pengecekan secara lanjut.
raymond pernah kejang tanpa didahului demam terlebih dahulu.
jika tidak tahu bilang maaf ners saya tidak tahu.
jawab dengan sopan, dan terlihat polos.
jawab sebagai pelaku utama.
puskesmas adalan sebuah klinik kesehatan.
jika ingin menjawab keluarga pertimbangkan kata kami.
jawablah dengan tepat dan tidak berlebihan.


Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)

#parameter tunning
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import langchain 
from langchain.chat_models import ChatOpenAI
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache


#from langchain.cache import InMemoryCache
#langchain.llm_cache = InMemoryCache()
#set_llm_cache(SQLiteCache(database_path=".raymond.db"))

llm = ChatOpenAI(streaming=True,model="gpt-4o",temperature=0.1
                 )

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


st.set_page_config(
    page_title="Pasien raymond",
    page_icon="👦🏻",
    layout="wide"
)


st.title("ChatGPT Clone")


# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, am ChatGPT clone"}
    ]


# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = rag_chain.invoke(user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
