from jinja2 import Template
from ollama import Client
import re
import json


from config import Config


_PROMPT_TEMPLATE = """Analysez l'avis suivant :
"{{text}}"
Déterminez si les aspects suivants sont exprimés et leur opinion correspondante : 
- Prix
- Cuisine
- Service
- Ambiance.
Utilisez ces valeurs : "Positive", "Négative", "Neutre", "NE". 
"NE" signifie que l'opinion n'est pas exprimée sur cet aspect.

Répondez uniquement en JSON :
{
  "Prix": "valeur",
  "Cuisine": "valeur",
  "Service": "valeur",
  "Ambiance": "valeur"
}
"""
# AVG MACRO ACC: 80.6 TOTAL EXEC TIME: 876

class LLMClassifier:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Instantiate an ollama client
        self.llmclient = Client(host=cfg.ollama_url)
        #self.model_name = 'llama3.1:latest'
        # self.model_name = 'gemma2:latest'
        self.model_name = 'gemma2:2b'
        self.model_options = {
            'num_predict': 500,  # max number of tokens to predict
            'temperature': 0.1,
            'top_p': 0.9,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)


    def predict(self, text: str) -> dict[str,str]:
        """
        Lance au LLM une requête contenant le texte de l'avis et les instructions pour extraire
        les opinions sur les aspects sous forme d'objet json
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 4 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        prompt = self.jtemplate.render(text=text)
        result = self.llmclient.generate(model=self.model_name, prompt=prompt, options=self.model_options)
        response = result['response']
        jresp = self.parse_json_response(response)
        return jresp

    def parse_json_response(self, response: str) -> dict[str, str] | None:
        m = re.findall(r"\{[^\{\}]+\}", response, re.DOTALL)
        if m:
            try:
                jresp = json.loads(m[0])
                for aspect, opinion in jresp.items():
                    if "non exprim" in opinion.lower():
                        jresp[aspect] = "NE"
                return jresp
            except:
                return None
        else:
            return None
