# BERTfluff

## BERT szókitalálos játék

Üres mondatokban hiányzik egy-egy szó, akár több mondatban ugyanaz; ezt tippeli és találja ki a BERT/FastText/más modell.

Ad egy mondatot a korpusz, az embernek és a BERT-nek is egy-egy tippje van. Rossz tipp után érkezik egy új mondat.
Minden mondatból ugyanaz a szó hiányzik, mindaddig, amíg valaki ki nem találja.

### Install

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```


A Gensim Guesserhez pedig a magyar modellek letöltése szükséges:

```bash
cd models
wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim
wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim.syn1neg.npy
wget https://nessie.ilab.sztaki.hu/~levai/hungarian_wv_models/hu_wv.gensim.wv.vectors.npy
```

### Használat

Az összes Guessernek van egy `make_guess` metódusa, amin keresztül egyik-másik modellel kommunikálni lehet.

#### `guessing_game.py`
Python interfész a játékhoz. Két játékot bonyolít le, feladni egy üres sztring megadásával lehet.

#### `flaskapp.py`
Flask szerveren elindítja a guessereket és a `test_requests.py` fájlban megadott módon lehet guesseket lekérni.

