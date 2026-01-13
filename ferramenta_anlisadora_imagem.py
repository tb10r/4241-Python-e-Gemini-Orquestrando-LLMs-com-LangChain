from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.globals import set_debug
from detalhes_imagem_modelo import DetalhesImagemModelo
import ast
import json

class FerramentaAnalisadoraImagem(BaseTool):
    name:str = "analisadoraimagem"
    description:str = """
    Utilize esta ferramenta sempre que for solicitado que você faça uma análise
    de imagem.

    # Entradas Requiridas
  - 'nome_imagem' (str) : Nome da imagem a ser analisada com extensão de JPG. Exemplo: teste.jpg ou teste.jpeg
    """

    return_direct: bool = True

    def _run(self, acao):
      # Normaliza `acao` para dict.
      if isinstance(acao, str):
          # tenta JSON primeiro (mais comum em integrações web)
          try:
              acao_dict = json.loads(acao)
          except Exception:
              try:
                  acao_dict = ast.literal_eval(acao)
              except Exception as e:
                  raise ValueError(f"Não foi possível interpretar 'acao': {e}")
      elif isinstance(acao, dict):
          acao_dict = acao
      else:
          raise ValueError(f"Tipo inválido para 'acao': {type(acao)}")

      caminho_imagem = acao_dict.get("nome_imagem", "")
      llm = ChatGoogleGenerativeAI(
            api_key = GEMINI_API_KEY,
            model = GEMINI_FLASH
      )

      imagem = encode_image(f"dados/{caminho_imagem}")
      template_analisador = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    assuma que você é um analisador de imagens. a sua tarefa principal consiste em: analisar uma imagem e extrair informaçôes importante de forma objetiva.

                    #FORMATO DE SAÍDA
                    Descrisão da imagem: 'coloque sua descrisão da imagem aqui'
                    Rótulos:'Coloque uam lista co três termos chaves separados por vírgula'

                    """
                ),
                (
                    "user",
                    [
                        {
                            "type" : "text",
                            "text" : "Descreva a imagem: "
                        },
                        {
                            "type" : "image_url",
                            "image_url" : {"url":"data:image/jpeg;base64,{imagem_analisada}"}
                        }
                    ]
                )
            ]
        )



      cadeia = template_analisador | llm | StrOutputParser()

      json_imagem = JsonOutputParser(
            pydantic_object=DetalhesImagemModelo
        )

      template_resposta = PromptTemplate(
            template = """
            Gere um resumo, utilizando uma linguagem clara e objetiva, focada no público brasileiro. A ideia é que a comunicação do resultado seja a mais facil possível, priorizando registros para consltas posteriores.

            #resultado da imagem
            {resposta_cadeia_analise_imagem}

            #formato de saida
            {formato_saida}

            """,
            input_variables = ["resposta_cadeia_analise_imagem"],
            partial_variables= {
                "formato_saida": json_imagem.get_format_instructions()
            }
        )
        

      cadeia_resumo = template_resposta | llm | json_imagem

      resposta_analise = cadeia.invoke({"imagem_analisada": imagem})

      resposta_resumo = cadeia_resumo.invoke({"resposta_cadeia_analise_imagem": resposta_analise})

      return resposta_resumo