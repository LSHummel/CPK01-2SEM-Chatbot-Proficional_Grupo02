# CPK01-2SEM-Chatbot-Proficional_Grupo02

## Memória

O projeto utiliza **Buffer Memory** por meio do `ConversationBufferMemory`.

A escolha do Buffer Memory permite manter o histórico completo da conversa durante a execução do chatbot. Dessa forma, o sistema consegue utilizar informações apresentadas anteriormente pelo usuário para manter o contexto dos próximos turnos.

A memória é criada no arquivo `memory_manager.py` e utilizada pela `ConversationChain` no arquivo `chains.py`.

### Limpeza da memória

Durante a execução, o usuário pode utilizar:

```text
limpar
```

para apagar o histórico atual da conversa.

A conversa também é encerrada utilizando:

```text
sair
```

Ao iniciar uma nova execução do programa, uma nova memória é criada.

---

## Execução


### 1. Instalar as dependências

```powershell
pip install -r requirements.txt
```

### 2. Configurar a Ollama Cloud

Crie um arquivo `.env` na raiz do projeto:

```text
CPK01-2SEM-Chatbot-Proficional_Grupo02/
├── .env
├── requirements.txt
└── app/
```

Dentro do `.env`, adicione sua chave:

```env
OLLAMA_API_KEY=sua_chave_da_ollama_cloud
```

A chave real da API não deve ser enviada ao GitHub.

### 3. Executar o chatbot

A execução deve ser realizada no terminal a partir da pasta raiz do projeto:

```powershell
python -m app.main
```

O chatbot será iniciado apresentando:

```text
--- Suporte Técnico de Aparelhos Eletrônicos ---
Digite 'sair' para encerrar.
Digite 'limpar' para apagar a memória.

Você:
```

---
