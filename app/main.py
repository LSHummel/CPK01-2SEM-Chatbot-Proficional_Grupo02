from pydantic import ValidationError

from .chains import chat_chain, analisar_atendimento
from .memory_manager import visualizar_memoria, limpar_memoria
from .rag_service import SuporteRAGService


def iniciar_chat():
    print("==================================================")
    print(" Suporte Técnico de Aparelhos Eletrônicos (CKP02)")
    print("==================================================")
    print("Comandos disponíveis:")
    print("  'rag: <sua dúvida>' -> Consulta direta aos Manuais Oficiais via RAG")
    print("  'memoria'           -> Visualiza histórico da conversa")
    print("  'limpar'            -> Limpa histórico da sessão")
    print("  'sair'              -> Encerra o programa\n")

    # Inicializa o serviço RAG modular com os manuais de suporte
    servico_rag = SuporteRAGService(docs_dir="docs", chunk_size=512)

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == "sair":
            break

        if pergunta.lower() == "limpar":
            limpar_memoria(chat_chain.memory)
            print("Memória limpa com sucesso.\n")
            continue

        if pergunta.lower() == "memoria":
            estado = visualizar_memoria(chat_chain.memory)
            print(estado)
            print()
            continue

        if not pergunta:
            print("Por favor, digite uma mensagem válida.\n")
            continue

        print("Processando...")

        # Se o usuário digitar 'rag:', direcionamos para o pipeline RAG do CKP02
        if pergunta.lower().startswith("rag:"):
            consulta_rag = pergunta[4:].strip()
            resposta_rag = servico_rag.buscar(consulta_rag)
            print(f"\n[Base de Manuais - RAG]:\n{resposta_rag}\n")
            continue

        # Fluxo normal conversacional do CKP01 com memória e saída estruturada
        resposta = chat_chain.predict(input=pergunta)
        print(f"Assistente: {resposta}\n")

        try:
            analise = analisar_atendimento(pergunta, resposta)
            print("Análise estruturada do atendimento:")
            print(analise.model_dump())
            print()
        except ValidationError as erro:
            print(f"Aviso de validação Pydantic: {erro}\n")
        except Exception as erro:
            print(f"Falha na análise estruturada: {erro}\n")


if __name__ == "__main__":
    iniciar_chat()