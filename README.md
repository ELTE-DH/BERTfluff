# BERTfluff

## BERT szókitalálos játék

Üres mondatokban hiányzik egy-egy szó, akár több mondatban ugyanaz; ezt tippeli és találja ki a BERT/FastText/más modell.

Ad egy mondatot a korpusz, az embernek és a BERT-nek is egy-egy tippje van. Rossz tipp után érkezik egy új mondat.
Minden mondatból ugyanaz a szó hiányzik, mindaddig, amíg valaki ki nem találja.

### Install

A `make` parancs futtatásával, vagy manuálisan az alábbi módon.

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Szükséges továbbá a Gensim magyar modell és a KenLM magyar modell:

```bash
mkdir -p models
wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim
wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim.syn1neg.npy
wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim.wv.vectors.npy
wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/10M_pruned.bin
```

### Használat

Az összes Guessernek van egy `make_guess` metódusa, amin keresztül egyik-másik modellel kommunikálni lehet.

### Példa kísérletek

Kétoldali, 10-széles növekvő kontextusok, egy körben egy-egy szót rak mindkét oldalhoz. 
100 kontextuson futtatja le a kísérletet BERT-tel és KenLM-mel.
```bash
python paper_resources/main.py --left-context_size 10 --right-context_size 10 \
                               --tactic rl --sample_size 100 --n_jobs 1 \
                               --server-addr "http://127.0.0.1:42069" \
                               --store_previous --freq-filename "resources/webcorp_2_freqs.tsv" \
                               --non-words "paper_resources/non_words.txt" \
                               --guesser bert kenlm
```

Kétoldali, 10-széles kontextusok, viszont minden KWIC-hez 12 kontextust gyűjt, 
és egy kísérlet abból áll, hogy hány küldönböző kontextus kell ahhoz, hogy a Guesser kitalálja a hiányzó szót. 

```bash
python paper_resources/main.py --left-context_size 10 --right-context_size 10 \
                               --tactic rl --sample_size 100 --n_jobs 1 \
                               --server-addr "http://127.0.0.1:42069" \
                               --store_previous --freq-filename "resources/webcorp_2_freqs.tsv" \
                               --multi_guess --multi_concord 12 --non-words "paper_resources/non_words.txt" \
                               --guesser bert kenlm
```

Szerver elindítása 32 workerrel (egy workernek 2 szál kell, a --threads 1 a Gunicorn parallelizmusára vonatkozik).
A timeoutot (-t) érdemes legalább 600-ra állítani, mert a BERT és KenLM számolások egy-egy szálat nézve lassúak.
```bash
cd src/bertfluff
gunicorn --workers=32 --threads=1 --worker-class=gthread \
         --chdir "${PWD}/../../" -t 900 -b 127.0.0.1:42069 \
         "flaskapp:create_app()" 
```
