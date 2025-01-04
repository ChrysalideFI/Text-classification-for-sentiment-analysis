from pandas import DataFrame
from tqdm import tqdm

from config import Config

# from llm_classifier import LLMClassifier
from PLMFT_Classifier import PLMFTClassifier

import re

def preprocess_text(text):
    # Enlever les emojis 
    text = re.sub(r'[^\w\s,.!?;:]', '', text)
    # Mettre en minuscules
    text = text.lower()
    return text

class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'PLMFT'
    # 'LLM'  # or 'PLMFT' (for Pretrained Language Model Fine-Tuning)

    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.classifier = PLMFTClassifier(cfg) # remplacer par le bon classifier : "LLMClassifier(cfg)" ou "PLMFTClassifier(cfg)"


    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        :param train_data:
        :param val_data:
        :param device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu
        :return:
        """
        for data in train_data:
            data['text'] = preprocess_text(data['text'])

        for data in val_data:
            data['text'] = preprocess_text(data['text'])
        # Mettre tout ce qui est nécessaire pour entrainer le modèle ici, sauf si methode=LLM en zéro-shot
        # auquel cas pas d'entrainement du tout
        pass



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        all_opinions = []
        preprocessed_texts = [preprocess_text(text) for text in texts]
        if self.METHOD == 'PLMFT':
            batch_predictions = self.classifier.predict(preprocessed_texts, device)
            all_opinions.extend(batch_predictions)
        else:
            for text in tqdm(preprocessed_texts):
                opinions = self.classifier.predict(text)
                all_opinions.append(opinions)
        return all_opinions
