from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from prompts import prompt


llm = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    temperature=0.7,
    num_predict=700,
)

parser = StrOutputParser()


chain = prompt | llm | parser





