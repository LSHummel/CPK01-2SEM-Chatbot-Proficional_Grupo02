# Projeto Chatbot Profissional · Suporte Técnico de Aparelhos Eletrônicos

Prompt Engineering & AI · FIAP · 2º Semestre 2026

## Integrantes

- Gabriel Camarosani Gouvea Gonçalves da Silva — RM 569189
- Gustavo Lima Andrade Santos — RM 571709
- Lucas Seiji Hummel — RM 569673
- Pedro Souza Castro — RM 569311
- Bruno Yudi Moritaka Kanashiro — RM 571776
- Lucas Barreto Santana — RM 573149

---

# CKP02 — RAG, Embeddings e Avaliação com ChromaDB

## Domínio e base de conhecimento

O domínio mantido para o CKP02 é suporte técnico de aparelhos eletrônicos. Os documentos utilizados pelo sistema ficam na pasta `docs/`.

Arquivos utilizados nos testes (resumos técnicos em `.txt` feitos a partir dos manuais originais em PDF):

- `dell-g15-5530-owners-manual-pt-br.txt`
- `manual-motorola-one.txt`
- `guia-rapido-roteador-mesh-v2.txt`
- `cpd60186.txt`
- `cv_mfc4620dw_epr_busr_leu359065.txt`

As fontes originais dos documentos devem ser registradas aqui:

| Documento | Produto | Fonte |
|---|---|---|
| `dell-g15-5530-owners-manual-pt-br.txt` | Dell G15 5530 | https://www.dell.com/support/product-details/pt-br/product/g-series-15-5530-laptop/resources/manuals |
| `manual-motorola-one.txt` | Motorola One | file:///C:/Users/user/Downloads/manual-motorola-one.pdf |
| `guia-rapido-roteador-mesh-v2.txt` | Positivo Smart Roteador Mesh | https://static.positivocasainteligente.com.br/manuais/2020/guia-rapido-roteador-mesh-v2.pdf |
| `cpd60186.txt` | Epson L3250 | https://files.support.epson.com/docid/cpd6/cpd60186.pdf |
| `cv_mfc4620dw_epr_busr_leu359065.txt` | Brother MFC-J4620DW | https://download.brother.com/welcome/doc100107/cv_mfc4620dw_epr_busr_leu359065.pdf |

## Pipeline RAG

O script `app/avaliar_rag.py` carrega os arquivos `.txt` de `docs/`, divide o conteúdo em trechos (chunks), cria embeddings e utiliza a busca vetorial para recuperar os trechos relacionados à pergunta.

O modelo gera uma resposta baseada nos trechos recuperados. Quando não encontra a informação nos documentos, deve informar que não encontrou a resposta na base.

Configuração: `OllamaEmbeddings(nomic-embed-text)`, `gemma4:cloud` com temperatura 0, `k=3` e overlap de `chunk_size // 8`.

## Avaliação com RAGAS

O script `app/avaliar_rag.py` compara duas configurações de `chunk_size`: 512 e 1024. O teste utiliza cinco perguntas, sendo quatro dentro dos manuais e uma sobre um assunto que não está nos documentos (geladeira), para testar a recusa.

As métricas utilizadas são:

- **Faithfulness:** avalia se a resposta está apoiada no contexto recuperado.
- **Answer relevancy:** avalia a relação entre a resposta e a pergunta.

## Resultados da execução

| chunk_size | Chunks gerados | Faithfulness | Answer relevancy |
|-----------:|---------------:|-------------:|-----------------:|
| 512        | 25             | 1,000        | 0,566            |
| 1024       | 11             | 1,000        | 0,688            |

## Análise dos resultados

As duas configurações obtiveram a mesma média de faithfulness: 1,000, ou seja, nenhuma resposta saiu do contexto recuperado. Na answer relevancy, `chunk_size = 1024` obteve 0,688 contra 0,566 de `chunk_size = 512`.

Com `chunk_size = 512`, a pergunta sobre o notebook Dell desligando sozinho não recuperou o trecho correto e foi respondida com "Não encontrei essa informação nos documentos fornecidos.". Com `chunk_size = 1024`, o trecho foi recuperado e a pergunta foi respondida.

A pergunta sobre o conserto do motor de uma geladeira está fora dos documentos e foi corretamente recusada nas duas configurações. O RAGAS atribui answer relevancy 0 para recusas, o que reduz a média. Considerando apenas as 4 perguntas dentro do domínio, as médias foram aproximadamente 0,71 (512) e 0,86 (1024).

Com base nesta execução, `chunk_size = 1024` apresentou melhor resultado. Os valores correspondem aos documentos e às perguntas configurados no script e podem variar levemente entre execuções.

---

# CKP01 — Chatbot Profissional

## Domínio

O chatbot atua no domínio de suporte técnico de aparelhos eletrônicos.

O objetivo é auxiliar o usuário na identificação de problemas comuns de hardware, software e configuração, fornecendo orientações práticas e seguras.

Usuários-alvo: pessoas que precisam de uma primeira orientação sobre problemas em aparelhos eletrônicos e querem entender possíveis causas e próximos passos.

O domínio foi mantido restrito para que o system prompt, a memória e a saída estruturada tenham uma finalidade clara. O mesmo domínio pode ser reutilizado nos próximos checkpoints do semestre.

## Arquitetura

O projeto segue a arquitetura de duas chains trabalhada na Aula 03:

```text
Pergunta do usuário
       |
       v
ConversationChain
+ ConversationBufferMemory
       |
       v
Resposta textual
       |
       +----------------------+
       |                      |
       v                      v
Histórico da sessão     Chain LCEL de análise
                        ChatPromptTemplate
                                |
                                v
                           ChatOllama
                                |
                                v
                      PydanticOutputParser
                                |
                                v
                        AnaliseAtendimento
```

### Chain 1 — conversa

`ConversationChain` recebe a pergunta e mantém o histórico em `ConversationBufferMemory`.

A memória é de curto prazo, existe durante a execução do programa e é apagada quando a aplicação é encerrada.

### Chain 2 — saída estruturada

A resposta produzida pelo chatbot é enviada para uma segunda chain:

```text
ChatPromptTemplate | ChatOllama | PydanticOutputParser
```

O resultado é um objeto `AnaliseAtendimento`, e não uma string livre.

## Justificativa da memória

Foi escolhido `ConversationBufferMemory` porque o caso de uso é um atendimento técnico por sessão, no qual as informações dos turnos anteriores podem ser importantes para entender sintomas, aparelho e tentativas já realizadas.

A principal vantagem é a fidelidade: o histórico completo fica disponível para os próximos turnos. A desvantagem é que o custo de tokens cresce conforme a conversa fica maior.

Essa escolha é adequada para uma sessão curta de suporte. Em uma conversa muito longa, uma estratégia Summary ou TokenBuffer poderia reduzir o crescimento do contexto.

## Demonstração de 5+ turnos

Após iniciar:

```bash
python -m app.main
```

use uma sequência semelhante:

```text
Você: Meu notebook está desligando sozinho.
Você: Isso acontece quando estou usando jogos.
Você: A temperatura parece alta.
Você: O problema começou ontem.
Você: Já limpei as entradas de ar.
Você: Qual era o problema que eu mencionei no início?
```

O último turno depende das informações anteriores e demonstra a memória conversacional.

Também é possível digitar:

```text
memoria
```

para visualizar o histórico atual, ou:

```text
limpar
```

para apagá-lo.

### Demonstração automática (sem digitação manual)

Para uma evidência objetiva e reprodutível do requisito "memória funciona em 5+ turnos", também é possível rodar:

```bash
python -m app.demo_memoria
```

Esse script executa 6 turnos fixos automaticamente (sem precisar digitar nada) e imprime, ao final, o estado bruto da memória via `load_memory_variables({})`, confirmando que o último turno — que pergunta "qual foi o aparelho mencionado no início?" — é respondido corretamente com base no primeiro turno da conversa.

## Pydantic v2 e validação

O schema `AnaliseAtendimento` possui seis campos:

- `categoria: str`
- `componente: str`
- `prioridade: Literal["baixa", "media", "alta"]`
- `resumo: str`
- `passos: List[str]`
- `precisa_tecnico: bool`

O `PydanticOutputParser` recebe esse schema e gera as instruções de formato que são inseridas no prompt por `.partial()`.

Depois da resposta do modelo, o parser valida os dados.

A aplicação também utiliza `try/except ValidationError` para evitar que uma saída inválida derrube o programa.

## Context Engineering

O system prompt foi organizado com XML tagging:

- `<persona>`
- `<dominio>`
- `<restricoes>`
- `<formato>`
- `<seguranca>`
- `<regra_final>`

As instruções críticas aparecem no início e no final do contexto. As mensagens do usuário são tratadas como dados e não podem substituir as regras do sistema.

O objetivo é aplicar o conceito de context engineering: selecionar e organizar intencionalmente o que entra na janela de contexto, evitando informação desnecessária.

## Context rot

O arquivo `app/context_rot.py` executa um experimento usando o mesmo modelo e a mesma pergunta final com diferentes tamanhos de contexto:

- 0 turnos
- 5 turnos
- 10 turnos
- 15 turnos
- 20 turnos

A instrução-alvo é colocada no meio do histórico e existem mensagens de distração ao redor dela.

O programa mede uma aproximação do número de tokens com `tiktoken` e registra se a resposta preservou a instrução-alvo.

Execute:

```bash
python -m app.context_rot
```

O resultado é gerado pelo modelo no momento da execução. Assim, a tabela apresentada pelo programa representa o teste executado naquele ambiente, em vez de uma pontuação inventada previamente.

`tiktoken` é utilizado como aproximação para a contagem, conforme apresentado na Aula 04; ele não representa necessariamente a contagem interna exata do `gemma4:cloud`.

## Meta prompting

O arquivo `app/metaprompting.py` aplica a técnica apresentada na Aula 04: o próprio modelo recebe o system prompt atual e produz uma versão otimizada.

Execute:

```bash
python -m app.metaprompting
```

A saída documenta o antes/depois de forma explícita:

- O `SYSTEM_PROMPT` original (o "antes"), com sua contagem aproximada de tokens via `tiktoken`.
- O prompt otimizado retornado pelo modelo entre `<prompt_otimizado>...</prompt_otimizado>` (o "depois"), seguido de 3 bullet points explicando as mudanças e da contagem de tokens da resposta.

Isso deixa o diferencial "meta prompting: usar o modelo para melhorar o prompt e documentar o antes/depois" evidenciado diretamente na execução, sem depender só da leitura do código.

## Segurança

O system prompt possui regras contra:

- prompt injection;
- jailbreak;
- tentativa de extração do system prompt;
- exposição de chaves e credenciais;
- instruções perigosas envolvendo hardware ou eletricidade.

A segurança foi implementada usando as técnicas trabalhadas no módulo: regras explícitas, separação entre dados e instruções e organização do contexto com XML.

---

# Estrutura do projeto

```text
CPK01-2SEM-Chatbot-Proficional_Grupo02/
├── .env
├── .env.example
├── README.md
├── requirements.txt
├── docs/
│   ├── dell-g15-5530-owners-manual-pt-br.txt
│   ├── manual-motorola-one.txt
│   ├── guia-rapido-roteador-mesh-v2.txt
│   ├── cpd60186.txt
│   └── cv_mfc4620dw_epr_busr_leu359065.txt
└── app/
    ├── __init__.py
    ├── main.py
    ├── chains.py
    ├── memory_manager.py
    ├── schemas.py
    ├── context_rot.py
    ├── metaprompting.py
    ├── demo_memoria.py
    ├── prompts.py
    └── avaliar_rag.py
```

# Como executar

### 1. Criar o ambiente

Na pasta raiz:

```bash
python -m venv .venv
```

Ative o ambiente virtual no Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar a Ollama Cloud

Crie `.env` a partir do exemplo:

```powershell
copy .env.example .env
```

Edite `.env`:

```text
OLLAMA_API_KEY=sua_chave_da_ollama_cloud
```

Nunca envie o `.env` para o GitHub ou para o arquivo `.zip`.

### 4. Executar o chatbot

```bash
python -m app.main
```

### 5. Executar a demonstração automática de memória (>= 5 turnos)

```bash
python -m app.demo_memoria
```

### 6. Executar o teste de context rot

```bash
python -m app.context_rot
```

### 7. Executar o meta prompting

```bash
python -m app.metaprompting
```

### 8. Executar a avaliação do RAG

No terminal do PyCharm, na pasta raiz do projeto:

```powershell
.\.venv312\Scripts\python.exe app/avaliar_rag.py
```

# Dependências

As dependências estão em `requirements.txt` e correspondem às bibliotecas utilizadas no projeto:

- LangChain Ollama
- LangChain Core
- LangChain Classic
- Pydantic v2
- python-dotenv
- tiktoken
- LangChain Community
- LangChain Text Splitters
- ChromaDB
- RAGAS

# Modelo utilizado

O projeto utiliza:

```text
gemma4:cloud
```

por meio da Ollama Cloud, com a chave armazenada em variável de ambiente.

Para os embeddings, o projeto utiliza `nomic-embed-text` com Ollama.
````

Pontos que mudei além do pedido:
- Removi a linha vazia e o texto solto do diagrama da arquitetura, mas o desenho é o mesmo.
- Atualizei os nomes dos arquivos de `docs/` e os números do RAGAS.

Antes de entregar, troque os "Inserir o link..." pelos links reais e confira o `.txt` do Dell e do Motorola contra os PDFs.