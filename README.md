# CKP01 — Chatbot Profissional · Suporte Técnico de Aparelhos Eletrônicos

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**

**Integrantes:**  
- Gabriel Camarosani Gouvea Gonçalves da Silva — RM 569189
- Gustavo Lima Andrade Santos — RM 571709 
- Lucas Seiji Hummel — RM 569673 
- Pedro Souza Castro — RM 569311
- Bruno Yudi Moritaka Kanashiro — RM 571776
- Lucas Barreto Santana — RM 573149

## Domínio

O chatbot atua no domínio de **suporte técnico de aparelhos eletrônicos**.

O objetivo é auxiliar o usuário na identificação de problemas comuns de hardware, software e configuração, fornecendo orientações práticas e seguras.

**Usuários-alvo:** pessoas que precisam de uma primeira orientação sobre problemas em aparelhos eletrônicos e querem entender possíveis causas e próximos passos.

O domínio foi mantido restrito para que o system prompt, a memória e a saída estruturada tenham uma finalidade clara. O mesmo domínio pode ser reutilizado nos próximos checkpoints do semestre.

## Requisitos atendidos

| Requisito | Status | Implementação |
|---|---|---|
| Pipeline LCEL | ✅ | `app/chains.py` usa `ChatPromptTemplate \| ChatOllama \| PydanticOutputParser` |
| ChatOllama | ✅ | `gemma4:cloud` via Ollama Cloud, usando `OLLAMA_API_KEY` no `.env` |
| ChatPromptTemplate | ✅ | `app/prompts.py`, com mensagens `system` e `human` e variáveis |
| Memória gerenciada | ✅ | `ConversationChain` + `ConversationBufferMemory` |
| Demonstração de memória | ✅ | `memoria` no terminal + roteiro de teste com 5+ turnos |
| Pydantic v2 | ✅ | `AnaliseAtendimento` em `schemas.py`, com 6 campos tipados |
| Validação | ✅ | `PydanticOutputParser` + `ValidationError` |
| Context engineering | ✅ | system prompt organizado com XML tagging |
| Context rot | ✅ | `app/context_rot.py`, com 0/5/10/15/20 turnos e contagem aproximada de tokens |
| Domínio documentado | ✅ | Esta seção + `SYSTEM_PROMPT` |
| Projeto local | ✅ | pacote `app/` + `.env.example` + `requirements.txt` + `README.md` |

### Diferenciais

| Diferencial | Status | Implementação |
|---|---|---|
| Métricas de contexto | ✅ | `tiktoken` mede tokens aproximados em `context_rot.py` |
| Meta prompting | ✅ | `app/meta_prompting.py` gera uma versão otimizada do system prompt |

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

A memória é de **curto prazo**, existe durante a execução do programa e é apagada quando a aplicação é encerrada.

### Chain 2 — saída estruturada

A resposta produzida pelo chatbot é enviada para uma segunda chain:

```text
ChatPromptTemplate | ChatOllama | PydanticOutputParser
```

O resultado é um objeto `AnaliseAtendimento`, e não uma string livre.

## Justificativa da memória

Foi escolhido **ConversationBufferMemory** porque o caso de uso é um atendimento técnico por sessão, no qual as informações dos turnos anteriores podem ser importantes para entender sintomas, aparelho e tentativas já realizadas.

A principal vantagem é a fidelidade: o histórico completo fica disponível para os próximos turnos. A desvantagem é que o custo de tokens cresce conforme a conversa fica maior.

Essa escolha é adequada para uma sessão curta de suporte. Em uma conversa muito longa, uma estratégia Summary ou TokenBuffer poderia reduzir o crescimento do contexto.

### Demonstração de 5+ turnos

Após iniciar:

```powershell
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

```text
<persona>
<dominio>
<restricoes>
<formato>
<seguranca>
<regra_final>
```

As instruções críticas aparecem no início e no final do contexto. As mensagens do usuário são tratadas como dados e não podem substituir as regras do sistema.

O objetivo é aplicar o conceito de **context engineering**: selecionar e organizar intencionalmente o que entra na janela de contexto, evitando informação desnecessária.

## Context rot

O arquivo `app/context_rot.py` executa um experimento usando o mesmo modelo e a mesma pergunta final com diferentes tamanhos de contexto:

```text
0 turnos
5 turnos
10 turnos
15 turnos
20 turnos
```

A instrução-alvo é colocada no meio do histórico e existem mensagens de distração ao redor dela.

O programa mede uma aproximação do número de tokens com `tiktoken` e registra se a resposta preservou a instrução-alvo.

Execute:

```powershell
python -m app.context_rot
```

O resultado é gerado pelo modelo no momento da execução. Assim, a tabela apresentada pelo programa representa o teste executado naquele ambiente, em vez de uma pontuação inventada previamente.

> `tiktoken` é utilizado como aproximação para a contagem, conforme apresentado na Aula 04; ele não representa necessariamente a contagem interna exata do `gemma4:cloud`.

## Meta prompting

O arquivo `app/meta_prompting.py` aplica a técnica apresentada na Aula 04: o próprio modelo recebe o system prompt atual e produz uma versão otimizada.

Execute:

```powershell
python -m app.meta_prompting
```

O resultado contém:

1. o prompt otimizado entre `<prompt_otimizado>...</prompt_otimizado>`;
2. uma explicação das principais mudanças.

Isso permite documentar o **antes/depois** do system prompt durante a apresentação ou avaliação.

## Segurança

O system prompt possui regras contra:

- prompt injection;
- jailbreak;
- tentativa de extração do system prompt;
- exposição de chaves e credenciais;
- instruções perigosas envolvendo hardware ou eletricidade.

A segurança foi implementada usando as técnicas trabalhadas no módulo: regras explícitas, separação entre dados e instruções e organização do contexto com XML.

## Estrutura do projeto

```text
CKP01_Aparelhos_Eletronicos_Grupo02/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── chains.py
│   ├── memory_manager.py
│   ├── schemas.py
│   ├── context_rot.py
│   ├── meta_prompting.py
│   └── prompts.py
├── .env.example
├── requirements.txt
└── README.md
```

## Como executar

### 1. Criar o ambiente

Na pasta raiz:

```powershell
python -m venv .venv
```

Ative o ambiente virtual no Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 3. Configurar a Ollama Cloud

Crie `.env` a partir do exemplo:

```powershell
copy .env.example .env
```

Edite `.env`:

```env
OLLAMA_API_KEY=sua_chave_da_ollama_cloud
```

**Nunca envie o `.env` para o GitHub ou para o arquivo `.zip`.**

### 4. Executar o chatbot

```powershell
python -m app.main
```

### 5. Executar o teste de context rot

```powershell
python -m app.context_rot
```

### 6. Executar o meta prompting

```powershell
python -m app.meta_prompting
```

## Dependências

As dependências estão em `requirements.txt` e correspondem às bibliotecas utilizadas nas aulas:

- LangChain Ollama
- LangChain Core
- LangChain Classic
- Pydantic v2
- python-dotenv
- tiktoken

## Modelo utilizado

O projeto utiliza exclusivamente:

```text
gemma4:cloud
```

por meio da Ollama Cloud, com a chave armazenada em variável de ambiente.

## Observações para a entrega

O arquivo `.env` não deve ser incluído na entrega.

A estrutura final para o `.zip` deve conter:

```text
app/
.env.example
requirements.txt
README.md
```

Antes de entregar, substituir os nomes e RMs dos integrantes no README e executar pelo menos uma vez:

```powershell
python -m app.main
```

para confirmar a configuração da chave e da Ollama Cloud.
