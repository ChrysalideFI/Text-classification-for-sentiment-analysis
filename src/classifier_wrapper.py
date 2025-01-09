from pandas import DataFrame
from tqdm import tqdm

from config import Config

# from llm_classifier import LLMClassifier
from PLMFT_Classifier import PLMFTClassifier
from llm_classifier import LLMClassifier

import re
import torch

def preprocess_text(Avis):
    # Enlever les emojis 
    Avis = re.sub(r'[^\w\s,.!?;:]', '', Avis)
    # Mettre en minuscules
    Avis = Avis.lower()
    return Avis

class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'LLM'
    # 'LLM'  # or 'PLMFT' (for Pretrained Language Model Fine-Tuning)

    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        # self.plm_name='SiddharthaM/hasoc19-bert-base-multilingual-cased-sentiment-new'
        self.cfg = cfg
        if self.METHOD == 'PLMFT':
            self.cfg.plm_name = "camembert-base"
            # 'SiddharthaM/hasoc19-bert-base-multilingual-cased-sentiment-new' #Modèle HuggingFace à utiliser
            self.classifier = PLMFTClassifier(cfg) # remplacer par le bon classifier : "LLMClassifier(cfg)" ou "PLMFTClassifier(cfg)"
        else:
            self.classifier = LLMClassifier(cfg)
 
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
            data['Avis'] = preprocess_text(data['Avis'])

        for data in val_data:
            data['Avis'] = preprocess_text(data['Avis'])
        
        # Mettre tout ce qui est nécessaire pour entrainer le modèle ici, sauf si methode=LLM en zéro-shot
        # auquel cas pas d'entrainement du tout

        if self.METHOD == 'PLMFT':
          self.classifier.train(train_data, val_data, device, epochs=3, batch_size=32, learning_rate=1e-5)  
                
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
        preprocessed_texts = [preprocess_text(Avis) for Avis in texts]
        if self.METHOD == 'PLMFT':
            batch_predictions = self.classifier.predict(preprocessed_texts, device)
            all_opinions.extend(batch_predictions)
        else:
            for Avis in tqdm(preprocessed_texts):
                opinions = self.classifier.predict(Avis)
                all_opinions.append(opinions)
        return all_opinions
