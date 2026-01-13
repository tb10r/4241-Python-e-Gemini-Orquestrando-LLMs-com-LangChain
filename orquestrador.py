
import warnings

# Suprime temporariamente FutureWarning gerado por pacotes deprecated
# (melhor solução: atualizar dependências para usar `google.genai`)
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from langchain.globals import set_debug
set_debug(False)

from langchain import hub
from langchain.agents import create_react_agent
from langchain.agents import tool
from ferramenta_anlisadora_imagem import FerramentaAnalisadoraImagem

class agenteorquestrador:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            api_key = GEMINI_API_KEY,
            model = GEMINI_FLASH
        )

        ferramenta_analisadora_imagem = FerramentaAnalisadoraImagem()

        # `tool` aceita (name_or_callable, runnable) positionalmente — não use
        # keywords como `name=` ou `func=`.
        self.tools = [
            tool(
                ferramenta_analisadora_imagem.name,
                ferramenta_analisadora_imagem.run,
                return_direct=ferramenta_analisadora_imagem.return_direct,
            )
        ]
    
        prompt = hub.pull("hwchase17/react")
        self.agente = create_react_agent(self.llm, self.tools, prompt)