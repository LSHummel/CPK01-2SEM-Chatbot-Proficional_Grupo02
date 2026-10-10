from pydantic import ValidationError

from .chains import chat_chain, analisar_atendimento
from .memory_manager import visualizar_memoria, limpar_memoria


def iniciar_chat():
    print("--- Suporte Técnico de Aparelhos Eletrônicos ---")
    print("Digite 'sair' para encerrar.")
    print("Digite 'limpar' para apagar a memória.")
    print("Digite 'memoria' para visualizar o histórico da sessão.\n")

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == "sair":
            break

        if pergunta.lower() == "limpar":
            limpar_memoria(chat_chain.memory)
            print("Memória limpa.\n")
            continue

        if pergunta.lower() == "memoria":
            estado = visualizar_memoria(chat_chain.memory)
            print(estado)
            print()
            continue

        if not pergunta:
            print("Digite uma pergunta.\n")
            continue

        print("Aguarde...")

        resposta = chat_chain.predict(input=pergunta)
        print(f"Assistente: {resposta}\n")

        # Demonstra a segunda chain: texto livre -> Pydantic validado.
        try:
            analise = analisar_atendimento(pergunta, resposta)
            print("Análise estruturada:")
            print(analise.model_dump())
            print()
        except ValidationError as erro:
            print("Não foi possível validar a análise estruturada.")
            print(erro)
            print()
        except Exception as erro:
            print(f"Falha na análise estruturada: {erro}\n")


if __name__ == "__main__":
    iniciar_chat()
