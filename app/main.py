from chains import chain

print("--- Suporte técnico de informatica ---")
pergunta = input("Informe o seu problema?\n")
print("Aguarde...")


resposta = chain.invoke({
    "pergunta": pergunta,
})

print(resposta)