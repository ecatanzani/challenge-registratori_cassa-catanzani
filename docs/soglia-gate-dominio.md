# La soglia del gate di dominio

Prima di interrogare l'LLM, il sistema misura quanto la domanda somiglia al pezzo di manuale più vicino. Se il punteggio è **sotto una soglia**, la domanda viene rifiutata subito, senza spendere una chiamata.

Questo serve per evitare di effettuare chiamate al modello quando non è necessario.

Il valore è in `.env` (`OFF_TOPIC_SIMILARITY_THRESHOLD`) e si ricalcola con:

```bash
python scripts/calibrate_threshold.py
```

Il valore della soglia è ottenuta con la seguente equazione:

```
soglia = min( minimo delle domande con risposta nel manuale − margine,
              minimo delle vicine non documentate)
```

Il margine non è fissato a mano: lo script lo ricava dai dati. Il procedimento e i
valori di oggi sono nella sezione [Estrazione dei valori puntuali](#estrazione-dei-valori-puntuali).

Il gate di dominio deve essre visto come **un'ottimizzazione di costo**: evita una chiamata LLM sulle domande palesemente estranee. 

Gli errori che questo gate può causare non sono simmetrici:

| errore | cosa succede | costo |
|---|---|---|
| **blocca una domanda legittima** | l'operatore, con la cassa ferma, si sente rispondere *"questa domanda non riguarda i manuali"* | aiuto negato: il fallimento peggiore del sistema |
| **lascia passare una domanda estranea** | una chiamata LLM in più, che finisce comunque in un rifiuto corretto grazie al grounding | qualche centesimo; la risposta all'utente resta giusta |

Questo gate non serve at attribuire la correttezza delle
risposte, che è invece il compito della funzione di **grounding** (`_validate_grounding()` in `chain.py`, che scarta ogni passo che citi un `chunk_id` non realmente recuperato).

## Le tre classi, e cosa il gate deve farne

L'eval set etichetta ogni domanda con un `kind`, e i tre valori hanno ruoli diversi nel calcolo.

**`rispondibile` — vincolo forte.** La risposta esiste nel manuale; bloccarle una nega assistenza.

**`vicina_non_documentata` — vincolo debole.** Argomento plausibile in un punto vendita ma assente da questo manuale: POS, lotteria scontrini, inventario. Bloccarle non sarebbe un risparmio neutro ma un **degrado**: il gate darebbe un
rifiuto generico con FAQ scollegate, mentre lasciandole passare l'LLM produce un rifiuto specifico con FAQ ricavate dai chunk davvero recuperati.

**`fuori_tema` — la misura, non un vincolo.** Nessun rapporto col dominio. Non entrano nel calcolo: sono l'unica classe su cui il gate ha un beneficio da dimostrare, e l'unica su cui misurarlo è metodologicamente pulito.

## Estrazione dei valori puntuali

Lo script `scripts/calibrate_threshold.py` ricava tutti i numeri della formula dall'eval set (`eval/eval_set.jsonl`), in quattro passi.

**1. Il punteggio di ogni domanda.** Per ciascuna delle 40 domande lo script calcola `compute_relevance_score()`: la similarità coseno fra la domanda, normalizzata come a runtime, e il chunk più vicino dell'indice. È lo stesso
numero che il gate confronta con la soglia quando l'app riceve una domanda.

Raggruppati per classe:

```
con risposta nel manuale : n=24  min 0.8372  max 0.8881
vicine non documentate   : n=10  min 0.8172  max 0.8530
fuori tema               : n=6   min 0.7637  max 0.8097
```

**2. Gli scarti, per stimare quanto una domanda nuova può cadere più in basso.**
Il minimo delle 24 domande con risposta dice solo quanto in basso arrivano quelle che conosciamo; una domanda legittima formulata diversamente può però fare peggio. Per stimare di quanto, lo script simula l'arrivo di domande nuove
(`calcola_scarti()`):

- sceglie a caso metà delle domande con risposta, 12 su 24, e ne prende il punteggio minimo come se fossero le uniche note;
- guarda le altre 12 e, per ognuna che cade **sotto** quel minimo, registra di quanto: è uno *scarto*;
- ripete 2000 volte, con seme fisso, così il risultato è riproducibile.

Metà dentro e metà fuori è il rapporto che mette più sotto sforzo la stima: con poche domande di calibrazione il minimo è meno rappresentativo, e gli scarti vengono più grandi.

Sul dataset attuale:

```
Scarti misurati su 1836 esclusioni: mediano 0.0031, massimo 0.0236
```

**3. Il margine.** Si prende lo scarto peggiore osservato e lo si moltiplica per
un fattore di sicurezza, `--interval`, che vale 1.5 se non specificato:

```
margine = 0.0236 × 1.5 = 0.0354
```

Il fattore è l'unica parte scelta e non misurata: tiene conto delle formulazioni che nell'eval set non compaiono.

**4. La soglia.** Si applica la formula, e lo script stampa quale dei due termini sta decidendo:

```
soglia = min(0.8372 − 0.0354 , 0.8172 − 0) = 0.8018   <- decide le domande con risposta nel manuale
```

Chiude con due controlli sulle stesse domande: legittime bloccate 0 su 34, fuori
tema fermate 5 su 6.

**Il valore in uso.** In `.env` la soglia è 0.8, un'aprossimazione del valore calcolato dal codice di calibrazione appena descritto.