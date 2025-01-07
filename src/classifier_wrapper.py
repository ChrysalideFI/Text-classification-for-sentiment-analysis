from pandas import DataFrame
from tqdm import tqdm

from config import Config

# from llm_classifier import LLMClassifier
from PLMFT_Classifier import PLMFTClassifier

import re
import torch

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
        # self.plm_name='SiddharthaM/hasoc19-bert-base-multilingual-cased-sentiment-new'
        self.cfg = cfg
        self.cfg.plm_name = 'SiddharthaM/hasoc19-bert-base-multilingual-cased-sentiment-new' #Modèle HuggingFace à utiliser
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

        if self.METHOD == 'PLMFT':
            # Sauvegarder les données d'entraînement dans un fichier temporaire
            # train_file_path = 'train_data.tsv'
            # train_df = DataFrame(train_data)
            # train_df.to_csv(train_file_path, sep='\t', index=False)

            # # Définir les aspects et les classes
            # aspects = ["Cuisine", "Ambiance", "Service", "Prix"]
            # classes = ["Positive", "Négative", "Neutre", "NE"]

            # # Définir le dispositif (GPU ou CPU)
            # device = torch.device(f'cuda:{device}' if device >= 0 else 'cpu')

            # # Entraîner le modèle
            # self.classifier.train(train_file_path, aspects, classes, device)
            # Définir les aspects et les classes

            #TEST CODE à enlever si besoin
            aspects = ["Cuisine", "Ambiance", "Service", "Prix"]
            classes = ["Positive", "Négative", "Neutre", "NE"]

            # Préparer les datasets
            train_dataset = self.prepare_data(train_data, aspects, classes)
            val_dataset = self.prepare_data(val_data, aspects, classes)

            # Créer un DataLoader pour l'ensemble de données
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Définir l'optimiseur
            optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=1e-5)

            # Boucle d'entraînement
            self.classifier.train()
            self.classifier.to(device)
            for epoch in range(3):
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                    outputs = self.classifier({'input_ids': input_ids, 'attention_mask': attention_mask})
                    loss = self.classifier.compute_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                print(f"Epoch {epoch + 1}/3, Loss: {total_loss / len(train_loader)}")

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
