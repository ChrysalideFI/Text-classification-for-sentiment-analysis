# I# Importation des bibliothèques
import torch
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import logging as hflogging  # gérer les logs et supprimer les avertissements inutiles

hflogging.set_verbosity_error()  # set logging to error to avoid warning messages

# Définition du modèle de classification multi-label
class TransformerMultiLabelClassifier(torch.nn.Module):
    def __init__(self, plm_name: str):
        super(TransformerMultiLabelClassifier, self).__init__()
        self.lmconfig = AutoConfig.from_pretrained(plm_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(plm_name)  # tokenisation
        self.lm = AutoModel.from_pretrained(plm_name, output_attentions=False)
        self.emb_dim = self.lmconfig.hidden_size
        self.output_size = 16  # 4 classes (Positive, Négative, Neutre et NE) pour chaque aspect (prix, cuisine, ambiance, service)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.emb_dim, self.output_size),
            torch.nn.Sigmoid()  
        )
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x):
        x = self.lm(x['input_ids'], x['attention_mask']).last_hidden_state
        global_vects = x.mean(dim=1)
        x = self.classifier(global_vects)
        return x

    def compute_loss(self, predictions, target):
        return self.loss_fn(predictions, target)

    def _format_predictions(self, predictions):
        aspects = ["Cuisine", "Ambiance", "Service", "Prix"]
        classes = ["Positive", "Négative", "Neutre", "NE"]
        formatted_predictions = []
        for pred in predictions:
            labels = {}
            for i, aspect in enumerate(aspects):
                aspect_scores = pred[i*4:(i+1)*4]
                max_score_idx = aspect_scores.index(max(aspect_scores))
                labels[aspect] = classes[max_score_idx]
            formatted_predictions.append(labels)
        return formatted_predictions

    def predict(self, texts, device, batch_size=32):
        self.eval()
        encodings = self.lmtokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        all_predictions = []
        device = torch.device(f'cuda:{device}' if device >= 0 else 'cpu')
        self.to(device)
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                outputs = self.forward({'input_ids': input_ids, 'attention_mask': attention_mask})
                formatted_outputs = self._format_predictions(outputs.cpu().numpy().tolist())
                all_predictions.extend(formatted_outputs)

        return all_predictions

# Nom du modèle pré-entraîné
plm_name = 'SiddharthaM/hasoc19-bert-base-multilingual-cased-sentiment-new'

# Initialiser le modèle
model = TransformerMultiLabelClassifier(plm_name)

# Préparation des données
def prepare_data(file_path, tokenizer, aspects, classes):
    # Charger les données depuis un fichier TSV
    df = pd.read_csv(file_path, delimiter='\t')

    # Fonction pour encoder les labels
    def encode_labels(row, aspects, classes):
        label_vector = []
        for aspect in aspects:
            for cls in classes:
                label_vector.append(1 if row[aspect] == cls else 0)
        return label_vector

    # Appliquer l'encodage des labels
    df['labels'] = df.apply(lambda row: encode_labels(row, aspects, classes), axis=1)

    # Tokeniser les textes
    encodings = tokenizer(df['Avis'].tolist(), truncation=True, padding=True, return_tensors="pt")

    # Créer les datasets PyTorch
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

        def __len__(self):
            return len(self.labels)

    # Créer les datasets
    labels = df['labels'].tolist()
    dataset = SentimentDataset(encodings, labels)
    return dataset

# Définir les aspects et les classes
aspects = ["Cuisine", "Ambiance", "Service", "Prix"]
classes = ["Positive", "Négative", "Neutre", "NE"]

print ("Data prepared successfully!")