import time
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset

# Verifica la versione di FAISS
print(faiss.__version__)

print(torch.__version__)  # Mostra la versione di PyTorch installata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Caricamento dataset
dataset_corpus = load_dataset("allenai/scifact", "corpus", trust_remote_code=True) 
dataset_claims = load_dataset("allenai/scifact", "claims", trust_remote_code=True)

# Filtra i claim con evidence_doc_id non vuoto su tutti gli split
dataset_claims = dataset_claims.filter(
    lambda doc_id: doc_id != "",
    input_columns=["evidence_doc_id"]
)

# Verifica che il filtraggio sia andato a buon fine
print("Dopo filtro:",{s: dataset_claims[s].num_rows for s in dataset_claims})
#print(dataset_claims['train']['cited_doc_ids'])

claims = dataset_claims['train']['claim'] + dataset_claims['validation']['claim']
abstracts = dataset_corpus['train']['abstract']
evidence_doc_ids = dataset_claims['train']['evidence_doc_id'] + dataset_claims['validation']['evidence_doc_id']
doc_ids = dataset_corpus['train']['doc_id']

#────────────────────────────────────────────────────────
# CONFRONTO MODELLI
#────────────────────────────────────────────────────────

K_values = [1,3,5,7,10]

model_names = [
    "thenlper/gte-large",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]

timings = {}

hit_k_results   = {m: [] for m in model_names}
mrr = {m: [] for m in model_names}

for model_name in model_names:

    print(f"\n Modello: {model_name}")

    model = SentenceTransformer(model_name, device = device)
    start_time = time.time()

    #Encoding
    claims_embeddings = model.encode(claims) #Tokenizzazione gestita da encode
    abstract_embeddings = model.encode(abstracts)

    #FAISS
    dim = claims_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(abstract_embeddings).astype('float32'))
    #Retrieval - FAISS
    k = 10  # Numero di vicini più simili che vogliamo recuperare
    D, I = index.search(np.array(claims_embeddings).astype('float32'), k) # D contiene le distanze e I contiene gli indici degli abstract più simili

    #Hit@K
    hits = {k: 0 for k in K_values}
    reciprocal_ranks = [] #MRR
    for i in range(len(claims)):
        ground_truth_doc_id = evidence_doc_ids[i]

        found_doc_ids = [str(doc_ids[I[i][j]]) for j in range(max(K_values))] 
        print(found_doc_ids)

        # Calcola Hit@K
        for k in K_values:
            if ground_truth_doc_id in found_doc_ids[:k]:
                hits[k] += 1

        # Calcola MRR
        if ground_truth_doc_id in found_doc_ids:
            rank = found_doc_ids.index(ground_truth_doc_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

        print(f"Doc_id più simile secondo il modello: {found_doc_ids[0]}")
        print(f"Ground Truth Document: {ground_truth_doc_id}")
        print(f"Claim: {claims[i]}")
        print(f"Abstract più simile: {abstracts[I[i][0]]}")  # Miglior vicino
        print(f"Similarità: {1/(1 + D[i][0])}")  # Similarità tra claim e abstract
        print("------")

    n = len(claims)
    hit_k_results[model_name] = [hits[k]/n for k in K_values]
    mrr[model_name] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    end_time = time.time()
    timings[model_name] = end_time - start_time
    print(f"Hit@K: {[f'{v:.3f}' for v in hit_k_results[model_name]]}")
    print(f"Tempo totale: {timings[model_name]:.1f}s")
    print(f"MRR: {mrr[model_name]:.4f}")

#────────────────────────────────────────────────────────
#Grafico Hit@K
#────────────────────────────────────────────────────────
plt.figure(figsize=(8,5))
for model_name, scores in hit_k_results.items():
    plt.plot(K_values, scores, marker='o', label=model_name)

plt.title("Confronto Hit@K tra modelli")
plt.xlabel("K")
plt.ylabel("Hit@K")
plt.xticks(K_values)
plt.ylim(0,1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────
# Grafico Tempi di Esecuzione
# ────────────────────────────────────────────────────────
plt.figure(figsize=(6,4))
plt.bar(timings.keys(), timings.values())
plt.title("Tempo di esecuzione totale per modello")
plt.ylabel("Secondi")
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#────────────────────────────────────────────────────────
# Grafico MRR
#────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.bar(mrr.keys(), mrr.values(), color='mediumseagreen')
plt.title("Confronto MRR tra modelli")
plt.xlabel("Modello")
plt.ylabel("MRR")
plt.ylim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#────────────────────────────────────────────────────────
# ScatterPlot MRR - ExecTime
#────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(timings.values(), mrr.values(), color='mediumseagreen', s=100)

for model in mrr.keys():
    plt.text(timings[model] + 0.5, mrr[model], model, fontsize=9)

plt.title("MRR vs Tempo di esecuzione")
plt.xlabel("Tempo di esecuzione (s)")
plt.ylabel("MRR")
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#────────────────────────────────────────────────────────
# ScatterPlot Hit@1 = Prec@1 - ExecTime
#────────────────────────────────────────────────────────
hit1 = {model: hit_k_results[model][0] for model in hit_k_results}

plt.figure(figsize=(8, 5))
plt.scatter(timings.values(), hit1.values(), color='orange', s=100)

for model in hit1:
    plt.text(timings[model] + 0.5, hit1[model], model, fontsize=9)

plt.title("Hit@1 vs Tempo di esecuzione")
plt.xlabel("Tempo di esecuzione (s)")
plt.ylabel("Hit@1")
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()