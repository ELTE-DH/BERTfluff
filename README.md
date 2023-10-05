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

# References

Ha ezt a programot használja, kérjük, hivatkozzon a következő cikkekre:

[Indig, B. and Lévai, D. (2022). __Okosabb vagy, mint egy XXXXXXXX? – egy nyelvi játéktól a nyelvmodellek összehasonlı́tásáig__. In Gábor Berend, et al., editors, _XVIII. Magyar Számı́tógépes Nyelvészeti Konferencia_, pages 31--44 Szeged, Hungary](https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/mszny2022.pdf)

```
@inproceedings{word-guessing-mszny2022,
    author = {Indig, Balázs and Lévai, Dániel},
    booktitle = {{XVIII}. {M}agyar {S}zámítógépes {N}yelvészeti {K}onferencia},
    title = {Okosabb vagy, mint egy {XXXXXXXX}? -- Egy nyelvi játéktól a nyelvmodellek összehasonlításáig},
    year = {2022},
    editor = {Gábor Berend and Gábor Gosztolya and Veronika Vincze},
    pages = {31--44},
    orcid-numbers = {Indig, Balázs/0000-0001-8090-3661}
}
```

[Balázs, I. and Dániel, L. (2023). __I’m Smarter than the Average BERT! – Testing Language Models Against Humans in a Word Guessing Game__. In Zygmunt Vetulani, et al., editors, _Human Language Technologies as a Challenge for Computer Science and Linguistics -- 2023_. pages 106–-110 Poznań, Poland](http://ltc.amu.edu.pl/)

```
@inproceedings{MTMT:33785196,
	author = {Indig, Balázs and Lévai, Dániel},
	booktitle = {Human Language Technologies as a Challenge for Computer Science and Linguistics -- 2023},
	title = {I’m Smarter than the Average BERT! – Testing Language Models Against Humans in a Word Guessing Game},
	year = {2023},
    editor = {Vetulani, Zygmunt and Paroubek, Patrick and Kubis, Marek},
	pages = {106-110},
	orcid-numbers = {Indig, Balázs/0000-0001-8090-3661}
}
```
