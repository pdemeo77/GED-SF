"""
Funzioni di Utilità per il Preprocessing dei Dataset di Graph Edit Distance
=============================================================================

Questo modulo fornisce funzioni di utilità per:
    - Caricare e preprocessare i dataset di grafi (AIDS, IMDB, Linux)
    - Costruire dataframe con i valori veri di GED (ground truth) da file JSON
    - Estrarre le etichette dei nodi per grafi etichettati
    - Convertire i dati dei grafi in formato CSV per un accesso efficiente

STRUTTURA DEI DATASET:
----------------------
I dataset sono memorizzati come file JSON nella cartella json_data/:
    - AIDS/train/*.json e AIDS/test/*.json (grafi molecolari con etichette)
    - IMDB/train/*.json e IMDB/test/*.json (grafi di collaborazione - senza etichette)
    - Linux/train/*.json e Linux/test/*.json (grafi di chiamate di sistema - senza etichette)

Ogni file JSON contiene:
    - 'graph': lista di archi [[nodo1, nodo2], [nodo3, nodo4], ...]
    - 'labels': lista di etichette dei nodi (solo per AIDS)

Il file TaGED.json contiene le distanze GED vere precomputate per coppie di grafi,
che servono come ground truth per valutare l'accuratezza del nostro metodo di approssimazione.
"""

import json
import networkx as nx
import pandas as pd
import os

def build_ged_ground_truth_dataframe(dataset_name, file_name='TaGED.json'):
    """
    Costruisce e salva un dataframe con i valori veri di GED da un file JSON.
    
    SCOPO:
    ------
    Il file TaGED.json contiene i valori di GED precalcolati per coppie di grafi,
    che servono come ground truth (verità di riferimento) per valutare l'accuratezza
    del nostro metodo di approssimazione basato su Gromov-Wasserstein.
    
    FORMATO JSON INPUT:
    -------------------
    Lista di elementi: [id_1, id_2, true_ged, ged_nc, ged_in, ged_ie, mappings]
    Dove:
        - id_1, id_2: ID dei due grafi da confrontare
        - true_ged: valore vero della Graph Edit Distance
        - ged_nc, ged_in, ged_ie: componenti del GED (node cost, insert, etc.)
        - mappings: mapping ottimale tra i nodi
    Noi utilizziamo solo id_1, id_2 e true_ged.
    
    Args:
        dataset_name (str): Nome del dataset ('AIDS', 'IMDB', o 'Linux')
        file_name (str): Nome del file JSON con i dati GED (default: 'TaGED.json')
    
    Output:
        Salva un file CSV in: True_GED/{dataset_name}/{dataset_name}_ged.csv
        con colonne: id_1, id_2, true_ged
    
    Esempio:
        >>> build_ged_ground_truth_dataframe('AIDS')
        Loaded 560 graph pairs for AIDS
        Saved to: True_GED/AIDS/AIDS_ged.csv
    """
    path = os.path.join('json_data', dataset_name, file_name)
    
    # Carica i dati veri di GED dal file JSON
    try:
        with open(path, 'r') as f:
            TaGED = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File non trovato in {path}")
        return
    
    # Estrai solo i campi rilevanti (ID dei grafi e GED vero)
    data = []
    for id_1, id_2, true_ged, ged_nc, ged_in, ged_ie, mappings in TaGED:
        data.append((id_1, id_2, true_ged))
    
    # Crea il DataFrame
    df = pd.DataFrame(data, columns=['id_1', 'id_2', 'true_ged'])
    print(f"Caricati {len(df)} coppie di grafi per {dataset_name}")
    print(df.head())
    print(10 * '*')
    
    # Salva in CSV
    output_folder = os.path.join("True_GED", dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{dataset_name}_ged.csv")
    df.to_csv(output_path, index=False)
    print(f"Salvato in: {output_path}\n")


def load_ged_ground_truth(dataset_name):
    """
    Carica e visualizza i dati veri di GED dal file CSV.
    
    SCOPO:
    ------
    Funzione di utilità per visualizzare rapidamente il contenuto del dataset GED
    e le sue statistiche (dimensione, range di valori, media).
    
    Args:
        dataset_name (str): Nome del dataset ('AIDS', 'IMDB', o 'Linux')
    
    Returns:
        None (stampa le informazioni del DataFrame sulla console)
    
    Esempio Output:
        AIDS Ground Truth GED Data:
           id_1  id_2  true_ged
        0   420   421        12
        1   420   422         8
        ...
        Shape: (560, 3) (graph pairs × features)
        GED range: [2, 24]
        Mean GED: 10.45
    """
    file_path = os.path.join("True_GED", dataset_name, f"{dataset_name}_ged.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    print(f"\n{dataset_name} Ground Truth GED Data:")
    print(df.head(15))
    print(f"Shape: {df.shape} (graph pairs × features)")
    print(f"GED range: [{df['true_ged'].min()}, {df['true_ged'].max()}]")
    print(f"Mean GED: {df['true_ged'].mean():.2f}\n")

def label_dict_construction():
    """
    Costruisce un dizionario che mappa gli ID dei grafi alle etichette dei nodi per il dataset AIDS.
    
    CONTESTO AIDS DATASET:
    ----------------------
    Il dataset AIDS contiene grafi molecolari dove ogni nodo rappresenta un atomo.
    Le etichette dei nodi indicano il tipo di atomo (C, O, N, S, etc.).
    Queste etichette sono memorizzate in file .onehot che contengono vettori one-hot
    che codificano le categorie degli atomi.
    
    FORMATO FILE .onehot:
    ---------------------
    Ogni file contiene una lista di vettori one-hot:
        [[1, 0, 0, 0],  # Nodo 0: tipo atomo 0 (es. Carbonio)
         [0, 1, 0, 0],  # Nodo 1: tipo atomo 1 (es. Ossigeno)
         [0, 1, 0, 0],  # Nodo 2: tipo atomo 1
         [1, 0, 0, 0]]  # Nodo 3: tipo atomo 0
    
    Questa funzione converte i vettori one-hot negli indici delle etichette:
        [1, 0, 0, 0] → 0
        [0, 1, 0, 0] → 1
        [0, 0, 1, 0] → 2
        etc.
    
    Returns:
        dict: Dizionario che mappa da graph_id (int) a lista di indici di etichette (list of int)
              Esempio: {42: [0, 2, 1, 0], 43: [1, 1, 3, 2], ...}
    
    Note:
        - Si applica solo al dataset AIDS (grafi molecolari con tipi di atomo)
        - Le etichette sono fondamentali per calcolare la distanza tra nodi nel nostro metodo
        - I file .onehot sono nella cartella json_data/AIDS/train/ e json_data/AIDS/test/
    """
    label_dict = {}
    
    # Processa sia il set di training che quello di test
    for type in ['train', 'test']:
        folder_path = os.path.join("json_data", "AIDS", type)
        
        try:
            # Trova tutti i file .onehot nella directory
            files = [file for file in os.listdir(folder_path) if file.endswith('.onehot')]
            
            for file in files:
                # Estrai l'ID del grafo dal nome del file (es. "42.onehot" → 42)
                graph_id = int(file.replace('.onehot', ''))
                
                # Carica le etichette codificate in one-hot
                with open(os.path.join(folder_path, file), 'r') as f:
                    content = json.load(f)
                    # Converti i vettori one-hot negli indici delle etichette
                    # [0,1,0,0] → 1, [1,0,0,0] → 0, [0,0,1,0] → 2, etc.
                    indexes = [sublist.index(1) for sublist in content]
                    label_dict[graph_id] = indexes
                    
        except FileNotFoundError:
            print(f"Attenzione: Cartella non trovata: {folder_path}")
    
    print(f"Caricate etichette per {len(label_dict)} grafi dal dataset AIDS")
    return label_dict

def save_graphs_on_df(dataset):
    """
    Carica i grafi dai file JSON e li salva in formato CSV per un accesso efficiente.
    
    PROBLEMA DA RISOLVERE:
    ----------------------
    I file JSON sono comodi per memorizzare i dati, ma inefficienti per l'accesso rapido.
    Durante gli esperimenti dobbiamo caricare migliaia di grafi ripetutamente.
    Questa funzione converte tutti i grafi in un unico file CSV che può essere caricato
    rapidamente con pandas.
    
    STRUTTURA OUTPUT CSV:
    ---------------------
    Il DataFrame generato ha le seguenti colonne:
        - graph_id: Identificatore unico del grafo (int)
        - graph_edge_list: Lista di archi serializzata come stringa JSON
        - node_labels: Etichette dei nodi (solo per AIDS)
    
    Esempio di riga:
        graph_id | graph_edge_list              | node_labels
        42       | "[[0,1],[1,2],[2,3]]"        | "['0', '1', '1', '0']"
    
    Args:
        dataset (str): Nome del dataset ('AIDS', 'IMDB', o 'Linux')
    
    Output:
        Salva un file CSV in: Dataset/{dataset}/{dataset}_graphs.csv
        
    Note:
        - Il dataset AIDS include le etichette dei nodi (tipi di atomo)
        - I dataset IMDB e Linux sono senza etichette (grafi non etichettati)
        - Le liste di archi sono memorizzate come stringhe JSON per compattezza
        - Se il file di output esiste già, salta l'elaborazione
    
    Esempio:
        >>> save_graphs_on_df('AIDS')
        Saved 560 graphs to: Dataset/AIDS/AIDS_graphs.csv
    """
    output_folder = os.path.join("Dataset", dataset)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{dataset}_graphs.csv")
    
    # Salta l'elaborazione se il file di output esiste già
    if os.path.exists(output_path):
        print(f"Il file di output esiste già per {dataset}: {output_path}. Elaborazione saltata.")
        return

    # Inizializza il DataFrame con le colonne appropriate
    if dataset == 'AIDS':
        graphs_df = pd.DataFrame(columns=['graph_id', 'graph_edge_list', 'node_labels'])
    else:
        graphs_df = pd.DataFrame(columns=['graph_id', 'graph_edge_list'])
    
    # Processa i set di training e test
    for type in ['train', 'test']:
        folder_path = os.path.join("json_data", dataset, type)
        
        try:
            # Trova tutti i file JSON dei grafi
            files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
            
            for file in files:
                # Estrai l'ID del grafo dal nome del file
                graph_id = int(file.replace('.json', ''))
                
                # Carica i dati del grafo
                with open(os.path.join(folder_path, file), 'r') as f:
                    content = json.load(f)
                    
                    if dataset != 'AIDS':
                        # Per i dataset senza etichette, controlla se esistono comunque
                        if "labels" in content:
                            node_labels = [str(label) for label in content["labels"]]
                            new_row = pd.DataFrame([{
                                'graph_id': graph_id, 
                                'graph_edge_list': content['graph'], 
                                'node_labels': node_labels
                            }])
                        else:
                            new_row = pd.DataFrame([{
                                'graph_id': graph_id, 
                                'graph_edge_list': content['graph']
                            }])
                        graphs_df = pd.concat([graphs_df, new_row], ignore_index=True)
                    else:
                        # Il dataset AIDS ha sempre le etichette
                        node_labels = [str(label) for label in content["labels"]]
                        new_row = pd.DataFrame([{
                            'graph_id': graph_id, 
                            'graph_edge_list': content['graph'], 
                            'node_labels': node_labels
                        }])
                        graphs_df = pd.concat([graphs_df, new_row], ignore_index=True)
                        
        except FileNotFoundError:
            print(f"Attenzione: Cartella non trovata: {folder_path}")
    
    # Salva in CSV
    graphs_df.to_csv(output_path, index=False)
    print(f"Salvati {len(graphs_df)} grafi in: {output_path}")

def load_graphs(dataset):
    """
    Carica i grafi dai file JSON in memoria (per test rapidi/debugging).
    
    UTILIZZO:
    ---------
    Questa funzione è utile per test rapidi e debugging, ma non è adatta per
    la produzione. Per un uso efficiente su larga scala, preferisci:
        save_graphs_on_df() + get_graph_by_id() da ged_computation.py
    
    LIMITAZIONE:
    ------------
    Per velocizzare i test, questa funzione carica solo i primi 2 file per directory.
    Non è pensata per caricare interi dataset.
    
    Args:
        dataset (str): Nome del dataset ('AIDS', 'IMDB', o 'Linux')
    
    Returns:
        dict: Dizionario che mappa da graph_id a:
              - nx.Graph per IMDB/Linux (grafi senza etichette)
              - tuple (nx.Graph, labels) per AIDS (grafi con etichette)
    
    Esempio:
        >>> graphs = load_graphs('AIDS')
        >>> graph_id = 42
        >>> G, labels = graphs[graph_id]
        >>> print(f"Il grafo {graph_id} ha {G.number_of_nodes()} nodi")
        >>> print(f"Etichette: {labels}")
    
    Note:
        - Carica solo i primi 2 file per directory (train e test) per test rapidi
        - Il dataset AIDS restituisce tuple (grafo, etichette)
        - Gli altri dataset restituiscono solo i grafi
    """
    graphs = {}
    
    for type in ['train', 'test']:
        print(f"Caricamento grafi {dataset} {type}...")
        folder_path = os.path.join("json_data", dataset, type)
        
        try:
            # Ottieni tutti i file JSON
            files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
            
            # Carica solo i primi 2 file per test rapidi
            first_two_files = files[:2]
            print(f"  Caricamento: {first_two_files}")
            
            for file in first_two_files:
                graph_id = int(file.replace('.json', ''))
                
                with open(os.path.join(folder_path, file), 'r') as f:
                    content = json.load(f)
                    edge_list = content['graph']
                    
                    # Costruisci il grafo NetworkX
                    G = nx.Graph()
                    G.add_edges_from(edge_list)
                    
                    if dataset != 'AIDS':
                        graphs[graph_id] = G
                    else:
                        # AIDS include le etichette dei nodi
                        labels = content['labels']
                        graphs[graph_id] = (G, labels)
                        
        except FileNotFoundError:
            print(f"Attenzione: Cartella non trovata: {folder_path}")
    
    print(f"Caricati {len(graphs)} grafi da {dataset}\n")
    return graphs


if __name__ == "__main__":
    """
    Esecuzione principale: costruisce i dataset di ground truth e grafi per tutti e tre i dataset.
    
    PIPELINE DI PREPROCESSING:
    --------------------------
    Questo passaggio di preprocessing dovrebbe essere eseguito una volta prima di condurre
    gli esperimenti. Converte i dati JSON grezzi in formato CSV per un accesso efficiente.
    
    STEPS:
    ------
    1. Costruisce i dataframe di ground truth GED da TaGED.json
       → True_GED/AIDS/AIDS_ged.csv
       → True_GED/IMDB/IMDB_ged.csv
       → True_GED/Linux/Linux_ged.csv
    
    2. Salva tutti i grafi in formato CSV
       → Dataset/AIDS/AIDS_graphs.csv
       → Dataset/IMDB/IMDB_graphs.csv
       → Dataset/Linux/Linux_graphs.csv
    
    UTILIZZO:
    ---------
    Esegui questo script direttamente:
        python utils.py
    
    Una volta completato, i file CSV saranno pronti per gli esperimenti.
    """
    print("="*60)
    print("Costruzione DataFrame di Ground Truth GED")
    print("="*60)
    
    # Costruisce i dataframe di ground truth GED
    build_ged_ground_truth_dataframe('AIDS')
    build_ged_ground_truth_dataframe('IMDB')
    build_ged_ground_truth_dataframe('Linux')
    
    print("\n" + "="*60)
    print("Salvataggio Grafi in DataFrame")
    print("="*60)
    
    # Salva i grafi in formato CSV
    save_graphs_on_df('AIDS')
    save_graphs_on_df('IMDB')
    save_graphs_on_df('Linux')
    
    print("\n" + "="*60)
    print("Preprocessing Completato!")
    print("="*60)

    
    


