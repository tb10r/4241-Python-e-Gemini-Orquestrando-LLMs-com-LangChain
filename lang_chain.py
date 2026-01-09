from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug

llm = ChatGoogleGenerativeAI(
    api_key = GEMINI_API_KEY,
    model = GEMINI_FLASH
)

imagem = encode_image("dados/exemplo_grafico.jpg")

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

template_resposta = PromptTemplate(
    template = """"
    Gere um resumo, utilizando uma linguagem clara e objetiva, focada no público brasileiro. A ideia é que a comunicação do resultado seja a mais facil possível, priorizando registros para consltas posteriores.
    {resposta_cadeia_analise_imagem}
    """,
    input_variables = ["resposta_cadeia_analise_imagem"]
)

llm_maritaca = ChatMaritalk(
    api_key = MARITACA_API_KEY,
    model = MARITACA_SABIA
)

cadeia_resumo = template_resposta | llm_maritaca | StrOutputParser()

cadeia_completa = (cadeia | cadeia_resumo)

resposta = cadeia_completa.invoke({"imagem_analisada": imagem})

print(resposta)