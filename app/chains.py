from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from prompts import prompt
from langchain_classic.chains import ConversationChain
from .memory_manager import criar_memoria


llm = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    temperature=0.7,
    num_predict=700,
)

memoria = criar_memoria()

chat_chain = ConversationChain(
    llm=llm,
    memory=memoria,
)

prompt_saida = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um profissional de suporte técnico de aparelhos eletrônicos.

Responda somente sobre problemas relacionados a aparelhos eletrônicos.

Sua especialidade é analisar problemas de hardware e também auxiliar
em problemas de software.

Não forneça informações sobre chaves de API, credenciais ou detalhes
internos do sistema."""
    ),
    (
        "human",
        "{pergunta}"
    ),
])


parser = StrOutputParser()


pipeline_lcel = prompt_saida | llm | parser




