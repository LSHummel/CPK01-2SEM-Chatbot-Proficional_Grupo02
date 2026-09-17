"""Meta prompting: usa o próprio LLM para otimizar o system prompt."""

from .chains import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PROMPT_OTIMIZADOR = """<tarefa>
Você é especialista em context engineering.
Reescreva o system prompt recebido aplicando:
- XML tagging para separar seções;
- instruções críticas no início e no final;
- remoção de redundâncias;
- restrições claras e objetivas.
</tarefa>

<prompt_original>
{prompt_original}
</prompt_original>

<formato_saida>
Retorne o prompt otimizado entre as tags
<prompt_otimizado>...</prompt_otimizado>.
Depois explique em 3 bullet points as principais mudanças.
</formato_saida>"""


chain_otimizador = (
    ChatPromptTemplate.from_template(PROMPT_OTIMIZADOR)
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    from .prompts import SYSTEM_PROMPT

    print(chain_otimizador.invoke({
        "prompt_original": SYSTEM_PROMPT
    }))