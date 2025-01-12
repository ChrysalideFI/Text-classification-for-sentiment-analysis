from jinja2 import Template
from ollama import Client
import re
import json


from config import Config


# _PROMPT_TEMPLATE = """Considérez l'avis suivant:

# "{{text}}"

# Quelle est la valeur de l'opinion exprimée sur chacun des aspects suivants : Prix, Cuisine, Service, Ambiance?

# La valeur d'une opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "Non exprimée".

# La réponse doit se limiter au format json suivant:
# { "Prix": opinion, "Cuisine": opinion, "Service": opinion, "Ambiance": opinion}."""
#AVG MACRO ACC: 66.67 TOTAL EXEC TIME: 719.0

# _PROMPT_TEMPLATE = """Considérez l'avis suivant :
# "{{text}}"
# Analysez et déterminez l'opinion exprimée sur les aspects suivants : 
# - Prix
# - Cuisine
# - Service
# - Ambiance
# Les valeurs possibles sont : "Positive", "Négative", "Neutre" ou "Non exprimée".
# Répondez uniquement en format JSON comme suit :
# {
#  "Prix": "valeur",
#  "Cuisine": "valeur",
#  "Service": "valeur",
#  "Ambiance": "valeur"
# }
# """
#AVG MACRO ACC: 78.34 TOTAL EXEC TIME: 568.2

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
# AVG MACRO ACC: 82.5 TOTAL EXEC TIME: 602.3

#_PROMPT_TEMPLATE = """Voici un exemple d’analyse :
# Avis : "La nourriture était délicieuse, mais le service était lent. Les prix sont abordables et l'ambiance chaleureuse."
# Résultat :
# {
#   "Prix": "Positive",
#   "Cuisine": "Positive",
#   "Service": "Négative",
#   "Ambiance": "Positive"
# }
# Maintenant, analysez cet avis :
# "{{text}}"
# Répondez en JSON :
# {
#   "Prix": "valeur",
#   "Cuisine": "valeur",
#   "Service": "valeur",
#   "Ambiance": "valeur"
# }
# """
# AVG MACRO ACC: 42.5 TOTAL EXEC TIME: 1172.5

# _PROMPT_TEMPLATE = """Considérez l'avis :
# "{{text}}"
# Donnez une opinion sur les aspects suivants (Prix, Cuisine, Service, Ambiance) avec une des valeurs : 
# "Positive", "Négative", "Neutre", "Non exprimée".
# Format de réponse strictement en JSON, maximum 4 lignes :
# {
#   "Prix": "valeur",
#   "Cuisine": "valeur",
#   "Service": "valeur",
#   "Ambiance": "valeur"
# }
# """
# AVG MACRO ACC: 58.34 TOTAL EXEC TIME: 553.5

# _PROMPT_TEMPLATE = """Voici un avis à analyser :
# "{{text}}"
# Pour chaque aspect suivant, indiquez l'opinion exprimée :
# - Prix
# - Cuisine
# - Service
# - Ambiance
# Utilisez uniquement l'une de ces valeurs : 
# - "Positive"
# - "Négative"
# - "Neutre"
# - "Non exprimée"
# Assurez-vous que chaque aspect a exactement une valeur. Répondez uniquement en JSON dans ce format strict :
# {
#   "Prix": "valeur",
#   "Cuisine": "valeur",
#   "Service": "valeur",
#   "Ambiance": "valeur"
# }
# """
# AVG MACRO ACC: 74.17 TOTAL EXEC TIME: 617.6

# _PROMPT_TEMPLATE = """Exemple d'analyse :
# Avis : "La nourriture est délicieuse, mais le service est lent."
# Réponse :
# {
#   "Prix": "Non exprimée",
#   "Cuisine": "Positive",
#   "Service": "Négative",
#   "Ambiance": "Non exprimée"
# }
# Analysez maintenant cet avis :
# "{{text}}"
# Répondez uniquement en JSON dans ce format strict :
# {
#   "Prix": "valeur",
#   "Cuisine": "valeur",
#   "Service": "valeur",
#   "Ambiance": "valeur"
# }
# """
# AVG MACRO ACC: 73.34 TOTAL EXEC TIME: 503.5

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
