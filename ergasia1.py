import pandas as pd
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split


# Βήμα 1ο: Φόρτωση και προετοιμασία του dataset SST-2
# Φόρτωση του dataset
dataset_path = "sst2-train.csv"
df = pd.read_csv(dataset_path)

# Εκτύπωση της δομής DataFrame για τον έλεγχο των ονομάτων των στηλών
print(df.head())

# Διαχωρισμός σε χαρακτηριστικά (κείμενα) και ετικέτες
text = df['sentence'].tolist()
label = df['label'].tolist()

# Διαχωρισμός του dataset σε train και validation set
train_text, validation_text, train_label, validation_label = train_test_split(text, label, test_size=0.2, random_state=42)


# Βήμα 2ο: Διανυσματοποίηση των κειμένων με Universal Sentence Encoder (USE)
# Φόρτωση του module Universal Sentence Encoder (USE)
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

# Ορισμός συνάρτησης για την ενσωμάτωση κειμένων
def embedding_text(text):
    embedding = model(text)
    return embedding

# Διανυσματικά train και validation κείμενα
train_embedding = embedding_text(train_text)
validation_embedding = embedding_text(validation_text)


# Εκτύπωση των 3 πρώτων embedding και των αντίστοιχων κειμένων
for i in range(3):
    print("Κριτική Ταινίας:", train_text[i])
    print("Embedding:", train_embedding[i])
    print("Βαθμολογία:", train_label[i])
    print()



import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# Βήμα 3ο: Εφαρμογή της μεθόδου t-SNE για μείωση της διαστατικότητας
# Εφαρμογή της t-SNE για τη μείωση των διαστάσεων των embedding
t_sne = TSNE(n_components=2, random_state=42)
train_embedding__t_sne = t_sne.fit_transform(train_embedding)


# Βήμα 4ο: Οπτικοποίηση Δεδομένων
# Οπτικοποίηση Δεδομένων από την t-SNE
plt.figure(figsize=(10, 6))
plt.scatter(train_embedding__t_sne[:, 0], train_embedding__t_sne[:, 1], c=train_label, cmap=plt.cm.get_cmap('RdYlGn', 2))
plt.title('t-SNE Οπτικοποίηση Δεδομένων με Universal Sentence Encoder Embeddings')
plt.xlabel('Συνιστώσα 1')
plt.ylabel('Συνιστώσα 2')
plt.colorbar(label='Ετικέτα')
plt.grid(True)
plt.show()