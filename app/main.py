from .memory_manager import visualizar_memoria, limpar_memoria
from .chains import chat_chain

print("--- Suporte técnico de informatica ---")
print("Digite 'sair' para encerrar.")
print("Digite 'limpar' para apagar a memória.\n")


while True:
    pergunta = input("Você: ")

    if pergunta.lower() == "sair":
        break

    if pergunta.lower() == "limpar":
        limpar_memoria(chat_chain.memory)
        print("Memória limpa.\n")
        continue

    print("Aguarde...")

    resposta = chat_chain.invoke({
        "input": pergunta
    })

    print(f"Assistente: {resposta['response']}\n")