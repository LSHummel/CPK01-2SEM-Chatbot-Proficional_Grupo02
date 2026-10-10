"""
Script de avaliação do CKP02 para rodar diretamente no PyCharm.
Compara duas configurações de chunk_size com 5 perguntas de teste.
Lê os arquivos .txt da pasta docs/.
"""
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Diretórios do projeto
PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DOCS = os.path.join(PASTA_PROJETO, "docs")
load_dotenv(os.path.join(PASTA_PROJETO, ".env"))

# Embeddings no Ollama local; respostas no Ollama Cloud
embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)
llm = ChatOllama(
    model="gemma4:cloud", temperature=0, base_url="https://ollama.com"
)

# Configuração do RAGAS (conforme material da Aula)
ragas_llm = LangchainLLMWrapper(llm)
ragas_embs = LangchainEmbeddingsWrapper(embeddings)
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embs

# 5 perguntas: 4 cobertas pelos manuais e 1 fora de domínio (para testar recusa)
PERGUNTAS = [
    "Como resolver o problema do notebook Dell desligando sozinho durante jogos?",
    "Como configurar a rede Wi-Fi do Positivo Smart Roteador Mesh?",
    "Como forçar o reinício do Motorola One se a tela travar?",
    "Como fazer a limpeza do cabeçote de impressão na impressora Epson L3250?",
    "Qual é o procedimento para consertar o motor de uma geladeira?"  # Pergunta fora do domínio
]

PROMPT_RAG = """<persona>
Você é um assistente técnico especializado no suporte de aparelhos eletrônicos.
</persona>

<instrucoes>
- Responda apenas com base nas informações do contexto abaixo.
- Cite o nome real do arquivo indicado no contexto: (fonte: nome do arquivo).
- Se não encontrar no contexto, responda: "Não encontrei essa informação nos documentos fornecidos."
- Não invente nada.
</instrucoes>

<contexto>
{contexto}
</contexto>

<pergunta>
{pergunta}
</pergunta>"""


def formatar(docs):
    return "\n\n".join([f"[{os.path.basename(d.metadata.get('source', 'doc'))}]\n{d.page_content}" for d in docs])


def testar_configuracao(chunk_size):
    print(f"\n=======================================================")
    print(f" TESTANDO CONFIGURAÇÃO: chunk_size = {chunk_size}")
    print(f"=======================================================")

    if not os.path.isdir(PASTA_DOCS):
        raise FileNotFoundError(f"Pasta docs não encontrada: {PASTA_DOCS}")

    # Carrega todos os arquivos .txt da pasta docs usando DirectoryLoader (como na aula)
    loader = DirectoryLoader(PASTA_DOCS, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    if not docs:
        raise ValueError("Nenhum arquivo .txt encontrado na pasta docs. Certifique-se de salvar os arquivos .txt lá dentro.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 8,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total de chunks gerados: {len(chunks)}")

    banco = Chroma.from_documents(chunks, embeddings, collection_name=f"teste_{chunk_size}")
    retriever = banco.as_retriever(search_kwargs={"k": 3})

    chain = ChatPromptTemplate.from_template(PROMPT_RAG) | llm | StrOutputParser()
    dados = {"question": [], "answer": [], "contexts": []}

    for i, p in enumerate(PERGUNTAS, 1):
        print(f"\n--- Pergunta {i} ---")
        print(f"Pergunta: {p}")
        trechos = retriever.invoke(p)
        resp = chain.invoke({"contexto": formatar(trechos), "pergunta": p})
        print(f"Resposta:\n{resp}")

        dados["question"].append(p)
        dados["answer"].append(resp)
        dados["contexts"].append([d.page_content for d in trechos])

    # Avaliação com RAGAS
    resultado = evaluate(
        Dataset.from_dict(dados),
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embs,
        raise_exceptions=True
    )
    tabela = resultado.to_pandas()
    print("\nRESULTADOS POR PERGUNTA:")
    print(tabela[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))

    banco.delete_collection()
    return {
        "chunk_size": chunk_size,
        "faithfulness": tabela["faithfulness"].mean(),
        "answer_relevancy": tabela["answer_relevancy"].mean()
    }


if __name__ == "__main__":
    resultados = []
    for tamanho in [512, 1024]:
        resultados.append(testar_configuracao(tamanho))

    print("\n=======================================================")
    print(" COMPARAÇÃO DAS MÉDIAS FINAIS")
    print("=======================================================")
    print("chunk_size | faithfulness | answer_relevancy")
    for r in resultados:
        print(f"{r['chunk_size']:10d} | {r['faithfulness']:.3f}        | {r['answer_relevancy']:.3f}")