import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# Configuração da Ollama Cloud conforme exigido no PDF
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
os.environ["OLLAMA_HOST"] = "https://ollama.com"
if OLLAMA_API_KEY:
    os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY

# Prompt estrito de grounding conforme a Aula 06 / Slide 12
PROMPT_RAG_TEXTO = """<persona>
Você é um assistente técnico especializado em suporte de aparelhos eletrônicos,
respondendo com base EXCLUSIVAMENTE nos manuais e documentos fornecidos.
</persona>

<instrucoes>
- Responda SOMENTE com informações do contexto abaixo.
- Sempre cite a fonte ao final: (fonte: {nome_doc}).
- Se a resposta não estiver no contexto, responda exatamente:
  "Não encontrei essa informação nos documentos fornecidos."
- Nunca invente, extrapole ou use conhecimento prévio externo.
</instrucoes>

<contexto>
{contexto}
</contexto>

<pergunta>
{pergunta}
</pergunta>"""


class SuporteRAGService:
    def __init__(self, docs_dir: str = "docs", chunk_size: int = 512):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_size // 8  # Conforme padrão da Aula 07 (Slide 17)

        # Modelo de embeddings fixado pelo professor
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # LLM fixado pelo professor: gemma4:cloud com temperature=0
        self.llm = ChatOllama(model="gemma4:cloud", temperature=0)

        self.retriever = None
        self.chain = None
        self._inicializar_pipeline()

    def _formatar_contexto(self, docs):
        """Formata os trechos recuperados citando a fonte."""
        partes = []
        for i, d in enumerate(docs, 1):
            fonte = d.metadata.get("source", "manual_tecnico")
            nome_limpo = os.path.basename(fonte)
            partes.append(f"[Trecho {i} — {nome_limpo}]\n{d.page_content}")
        return "\n\n---\n\n".join(partes)

    def _inicializar_pipeline(self):
        """Executa load -> split -> embed -> store -> retrieve -> chain."""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            return

        loader = DirectoryLoader(self.docs_dir, glob="*.txt", loader_cls=TextLoader)
        documentos = loader.load()
        if not documentos:
            return

        # Splitter com os separadores obrigatórios do PDF do professor
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documentos)

        # ChromaDB local com coleção identificando o domínio
        nome_colecao = f"suporte_eletronicos_{self.chunk_size}"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=nome_colecao
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(PROMPT_RAG_TEXTO)

        self.chain = (
                {
                    "contexto": self.retriever | RunnableLambda(self._formatar_contexto),
                    "pergunta": RunnablePassthrough(),
                    "nome_doc": RunnableLambda(lambda _: "Manuais Técnicos Oficiais")
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def buscar(self, consulta: str) -> str:
        """Função modular requisitada pelo professor para reaproveitamento no CKP03."""
        if not self.chain:
            return "Base de conhecimento não indexada ou vazia."
        return self.chain.invoke(consulta)