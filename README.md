
# Projet d'Analyse de sentiment

Nous avons analyser l'opinion exprimée sur des avis de restaurant sur TripAdvisor. Pour se faire, nous avons d'abord opter pour l'approche de fine-tuning qui s'est avéré non concluante dans les temps imparties. 
Nous avons donc choisi l'option du LLMClassifieur avec comme modèle le gemma2:2b. 

# Présentation du classifieur
Le classifieur LLMClassifier utilise un modèle de langage pré-entraîné (LLM) pour analyser les avis textuels et déterminer les opinions exprimées sur quatre aspects spécifiques : Prix, Cuisine, Service, et Ambiance. Il utilise des représentations textuelles basées sur des modèles de langage avancés. Le texte de l'avis est intégré dans un modèle de langage pré-entraîné, qui génère des représentations vectorielles des séquences de texte. Ces représentations sont ensuite utilisées pour prédire les opinions sur les différents aspects. L'architecture du classifieur repose sur un modèle de langage pré-entraîné, spécifiquement configuré pour la tâche de classification de texte. Le modèle utilisé est gemma2:2b. Le modèle est configuré pour générer des prédictions sous forme de JSON, indiquant les opinions sur les aspects spécifiés. Autre que les ressources précédemment mentionnés, nous nous somme servis de Ollama.

## Hyper-paramètres
Les principaux hyper-paramètres utilisés dans le classifieur sont les suivants :

* num_predict : 500 (nombre maximum de tokens à prédire)
* temperature : 0.1 (paramètre de température pour le contrôle de la diversité des prédictions)
* top_p : 0.9 (paramètre de filtrage pour le contrôle de la diversité des prédictions)

## Tentatives
Nous avons tentés de nombreux prompt et comparé leur taux d'accuracy moyen. Voici nos différents essais avec leur taux d'accuracy moyen à la suite : 
_PROMPT_TEMPLATE = """Considérez l'avis suivant:

"{{text}}"

Quelle est la valeur de l'opinion exprimée sur chacun des aspects suivants : Prix, Cuisine, Service, Ambiance?

La valeur d'une opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "Non exprimée".

La réponse doit se limiter au format json suivant:
{ "Prix": opinion, "Cuisine": opinion, "Service": opinion, "Ambiance": opinion}."""

AVG MACRO ACC: 66.67 

_PROMPT_TEMPLATE = """Considérez l'avis suivant :
 "{{text}}"
 Analysez et déterminez l'opinion exprimée sur les aspects suivants : 
 - Prix
 - Cuisine
 - Service
 - Ambiance
 Les valeurs possibles sont : "Positive", "Négative", "Neutre" ou "NE". 
"NE" signifie que l'opinion n'est pas exprimée sur cet aspect.
 Répondez uniquement en format JSON comme suit :
 {
  "Prix": "valeur",
  "Cuisine": "valeur",
  "Service": "valeur",
  "Ambiance": "valeur"
 }
 """

AVG MACRO ACC: 78.34

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

AVG MACRO ACC: 82.5

_PROMPT_TEMPLATE = """Voici un exemple d’analyse :
Avis : "La nourriture était délicieuse, mais le service était lent. Les prix sont abordables et l'ambiance chaleureuse."
Résultat :
{
  "Prix": "Positive",
  "Cuisine": "Positive",
  "Service": "Négative",
  "Ambiance": "Positive"
}
Maintenant, analysez cet avis :
"{{text}}"
Répondez en JSON :
{
  "Prix": "valeur",
  "Cuisine": "valeur",
  "Service": "valeur",
  "Ambiance": "valeur"
}
"""

AVG MACRO ACC: 42.5

_PROMPT_TEMPLATE = """Considérez l'avis :
"{{text}}"
Donnez une opinion sur les aspects suivants (Prix, Cuisine, Service, Ambiance) avec une des valeurs : 
"Positive", "Négative", "Neutre", "Non exprimée".
Format de réponse strictement en JSON, maximum 4 lignes :
{
  "Prix": "valeur",
  "Cuisine": "valeur",
  "Service": "valeur",
  "Ambiance": "valeur"
}
"""

AVG MACRO ACC: 58.34

_PROMPT_TEMPLATE = """Voici un avis à analyser :
"{{text}}"
Pour chaque aspect suivant, indiquez l'opinion exprimée :
- Prix
- Cuisine
- Service
- Ambiance
Utilisez uniquement l'une de ces valeurs : 
- "Positive"
- "Négative"
- "Neutre"
- "Non exprimée"
Assurez-vous que chaque aspect a exactement une valeur. Répondez uniquement en JSON dans ce format strict :
{
  "Prix": "valeur",
  "Cuisine": "valeur",
  "Service": "valeur",
  "Ambiance": "valeur"
}
"""

AVG MACRO ACC: 74.17

_PROMPT_TEMPLATE = """Exemple d'analyse :
Avis : "La nourriture est délicieuse, mais le service est lent."
Réponse :
{
  "Prix": "Non exprimée",
  "Cuisine": "Positive",
  "Service": "Négative",
  "Ambiance": "Non exprimée"
}
Analysez maintenant cet avis :
"{{text}}"
Répondez uniquement en JSON dans ce format strict :
{
  "Prix": "valeur",
  "Cuisine": "valeur",
  "Service": "valeur",
  "Ambiance": "valeur"
}
"""

AVG MACRO ACC: 73.34 

## Résultats
Nous avons opté pour un prompt qui à atteint 80.6% d'accuracy moyenne en 876s. Non seulement il s'agissait de notre prompt le plus précis mais aussi le plus rapide (le second prompt le plus précis a pris 936s). Voici le prompt sélectionné : 

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

    
## Auteurs

- [@Juliette012](https://www.github.com/Juliette012) Juliette Dartois
- [@ChrysalideFI](https://www.github.com/ChrysalideFI) Montana Katz

