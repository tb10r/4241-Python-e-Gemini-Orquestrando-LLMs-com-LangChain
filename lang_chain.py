from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.globals import set_debug
from detalhes_imagem_modelo import DetalhesImagemModelo

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

print(resposta_resumo)