import torch
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as hflogging  # gérer les logs et supprimer les avertissements inutiles

hflogging.set_verbosity_error()  # set logging to error to avoid warning messages

# Définition du modèle de classification multi-label
class PLMFTClassifier(torch.nn.Module):
    
    def __init__(self, cfg):
        super(PLMFTClassifier, self).__init__()
        self.plm_name = cfg.plm_name
        self.lmconfig = AutoConfig.from_pretrained(self.plm_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(self.plm_name)  # tokenisation
        self.lm = AutoModelForSequenceClassification.from_pretrained(self.plm_name, output_attentions=False)
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

    # def train(self, train_data, val_data, device): 
    def train(self, train_data, val_data, device, epochs=3, batch_size=32, learning_rate=1e-5):
        # Charger les données depuis les listes de dictionnaires
        df_train = pd.DataFrame(train_data)
        df_val = pd.DataFrame(val_data)
        
        # Tokeniser les textes
        train_encodings = self.lmtokenizer(
            df_train['Avis'].tolist(),
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        val_encodings = self.lmtokenizer(
            df_val['Avis'].tolist(),
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
         # Extraire les labels
        train_labels = df_train[['Prix', 'Cuisine', 'Service', 'Ambiance']].values
        val_labels = df_val[['Prix', 'Cuisine', 'Service', 'Ambiance']].values

        # Créer les datasets PyTorch
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = torch.tensor(labels, dtype=torch.float)

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)

        # Créer les datasets
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)

        # Met le modèle en mode entraînement et le déplacer vers le device spécifié (CPU ou GPU)
        self.train()
        self.to(device)

        # Création des dataloaders afin de charger les données en mini-batchs
        # Ils permettent de mélanger les données et faire le chargement en parallèle ce qui devrait améliorer l'efficacité de l'entraînement
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        #Optimiseur
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        # Boucle d'entraînement
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = self.forward({'input_ids': input_ids, 'attention_mask': attention_mask})
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
                