# Fusione RRF e reranker

Il retrieval passa da quattro stadi, ognuno con un punteggio proprio. Solo l'ultimo arriva all'utente, e capire perché richiede di vedere cosa fa ciascuno.

## La pipeline e i suoi quattro punteggi

```
domanda
   │
   ├──► retriever denso  ──► 8 candidati per similarità coseno     ① score in [-1, 1]
   │
   ├──► BM25             ──► 8 candidati per punteggio lessicale   ② ecore illimitato
   │
   ▼
fusione RRF  ──► combina per POSIZIONE, tiene i primi 5            ③ score ~0,01
   │
   ▼
cross-encoder ──► rilegge ogni coppia (domanda, chunk)             ④ P(pertinente) e la riordina
   │            
   ▼
LLM riceve i 5 chunk
```

| punteggio | chi lo produce | cosa misura | dove si usa |
|---|---|---|---|
| ① coseno | retriever denso | vicinanza semantica fra due vettori | ordinamento denso **e gate di dominio** |
| ② BM25 | retriever lessicale | sovrapposizione di parole, pesata per rarità | solo ordinamento lessicale |
| ③ RRF | fusione | quanto in alto sta un chunk nelle due classifiche | scelta dei 5 candidati |
| ④ cross-encoder | reranker | probabilità stimata che il chunk risponda | ordine finale, **pannello della UI** |

① e ② non escono mai dalla fase di retrieval, con un'eccezione: il coseno è anche il segnale del gate di dominio, perché è l'unico dei due su una scala fissa (vedi [soglia-gate-dominio.md](soglia-gate-dominio.md)).

## Perché RRF: due scale incompatibili

Il coseno vive tra −1 ed 1. Lo score ottenuto da BM25, invece, non ha limite superiore e dipende da quante parole ha la domanda e da quanto sono rare nel corpus: la stessa domanda sul rotolo di carta vale 57 detta in una parola e 72 detta in una frase. Sommare gli score del retriever denso e semantico non ha senso, mentre normalizzarli introduce ipotesi che nessuno dei due soddisfa.

La soluzione è l'utilizzo del **Reciprocal Rank Fusion**, che aggira il problema ignorando i punteggi e usando solo le posizioni: ogni chunk riceve `peso / (60 + posizione)` da ciascuna classifica in cui compare, e i contributi si sommano. Un chunk che entrambi i retriever mettono in alto vince; uno che solo uno dei due vede, conta meno.

RRF **non legge il testo**, ma è un'aggregazione di opinioni già date: se entrambi i retriever mettono in basso il chunk giusto, RRF non ha modo di accorgersene.

## Perché un secondo stadio: bi-encoder e cross-encoder

Il retriever denso è un **bi-encoder**: domanda e chunk vengono trasformati in vettori *separatamente*, e poi confrontati con un prodotto scalare. I chunk si codificano una volta sola durante l'ingestion, quindi la ricerca è istantanea, ma i due testi non si vedono mai, e tutta la loro interazione si riduce a un numero finale.

Il reranker è invece un **cross-encoder**: domanda e chunk vengono concatenati in un'unica sequenza de attraversano il transformer *insieme*. Ogni parola della domanda può fare `attenzione` su ogni parola del chunk, strato dopo strato: il modello legge la coppia come la leggerebbe una persona, chiedendosi se quel paragrafo risponde a quella domanda.

Il reranker è molto più accurato, ma non si può precalcolare: serve un'esecuzione completa del modello per ogni coppia domanda-risposta. Su tutti i chunk sarebbe troppo lento; sui milgiori 5 costa invece circa 80 ms. Da qui i due stadi: il primo scarta rapidamente quasi tutto il manuale, il secondo giudica con cura ciò che resta.

## Cos'è il punteggio del reranker

Il cross-encoder usato (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) ha **un solo output** (`num_labels = 1`); è un classificatore binario di rilevanza. Il modello produce in output un valore (logit) `z` tale che `σ(z)` (funzione sigmoide) stimi la probabilità che la coppia sia pertinente.

### Perché la sigmoide e non una normalizzazione min-max

Una normalizzazione di tipo *min-max*, sui cinque punteggi mostrati rimappa sempre il migliore a 1. 

Sulla domanda *"Come programmo l'intestazione dello scontrino?"*, dove il retrieval non aveva trovato il chunk giusto:

```
logit     [-1,39  -1,44  -1,77  -1,90  -3,12]
min-max   [ 1,00   0,97   0,78   0,70   0,00]    barra piena: sembra un centro
sigmoide  [ 0,20   0,19   0,14   0,13   0,04]    tutto basso: nessuno convince
```

L'algoritmo *min-max* mente proprio nel caso che conta, mentre la sigmoide conserva l'informazione assoluta, che è ciò che un pannello chiamato *"Perché questa risposta?"* deve comunicare.

Un limite, da dichiarare: `σ(z)` è la probabilità **stimata dal modello** sotto la distribuzione su cui è stato addestrato (MS MARCO, passaggi web tradotti). Il modello non è calibrato su manuali di registratori di cassa: un 0,20 non significa quindi che il 20% di quei chunk sia davvero pertinente. Per questo motivo nella UI è chiamata
*rilevanza stimata* e non probabilità.

## Serve davvero il reranker, se RRF basta a scegliere i 5?

Ho effetturato i test su 48 domande (le 24 rispondibili dell'eval set più 24 costruite da sezioni del manuale che l'eval set non copre):

| | hit@1 | hit@5 | MRR |
|---|---|---|---|
| solo RRF | 35/48 | 48/48 | 0,846 |
| RRF + reranker | **41/48** | 48/48 | **0,917** |

Il reranker sposta la risposta giusta in 15 domande su 48: più in alto in 11, più in basso in 4. Sei risposte in più finiscono in prima posizione.

Ma **hit@5 è identico**, e non per caso: il reranker agisce sui 5 chunk che RRF ha già scelto, quindi non può aggiungerne né toglierne. Può solo cambiarne l'ordine, e l'LLM riceve comunque tutti e 5 i chunk.

Il reranker resta per tre ragioni:

- **l'ordine nel contesto**: i modelli linguistici non pesano tutte le posizioni allo stesso modo, e il chunk giusto in cima è preferibile;
- **il punteggio mostrato**: il rerank permette di assegnare punteggi di rilevanza ad ogni chunk;
- **la robustezza su domande più difficili**: RRF funziona finché almeno uno dei due retriever vede giusto; il reranker vale di più dove entrambi sbagliano.

## Perché non allargare il gruppo di candidati

La configurazione classica di un sistema a due stadi fa recuperare al primo stadio più candidati di quanti ne servono, e lascia scegliere al secondo. È stato misurato l'impatto di questo processo, facendo produrre ad RRF N candidati e tenendone solo 5 dopo il rerank:

| candidati da RRF | hit@1 | hit@5 | MRR | tempo del rerank |
|---|---|---|---|---|
| **5 (in uso)** | 41/48 | 48/48 | **0,917** | 84 ms |
| 8 | 41/48 | 48/48 | 0,915 | 130 ms |
| 10 | 41/48 | 48/48 | 0,913 | 162 ms |
| 12 | 41/48 | 48/48 | 0,912 | 188 ms |
| 16 | 41/48 | 48/48 | 0,911 | 203 ms |

Nessuna domanda migliora; allargando, una o due peggiorano.

Il vantaggio di un gruppo più largo dipende da una sola quantità: quante volte la risposta giusta sta fra i candidati allargati ma non fra i primi 5. Solo in quei casi il reranker ha qualcosa da recuperare. Sui dati attuali quel numero è zero, perché RRF mette già la pagina giusta fra le prime cinque ovunque: allargando entrano solo distrattori, e ogni tanto il reranker ne preferisce uno a quello giusto. Un secondo stadio può correggere gli errori del primo; se il primo non ne fa, può solo introdurne.

Il criterio per rivalutare la scelta è misurabile:

```
recall@5 di RRF  <  recall@10 di RRF   ->   allargare il gruppo può aiutare
recall@5 di RRF  =  recall@10 di RRF   ->   si aggiungono solo distrattori
```

Due situazioni lo renderebbero probabile: l'indicizzazione di altri manuali — più modelli di cassa significano molti chunk quasi identici — e domande reali più difficili di quelle dell'eval set, scritte da chi ha costruito il sistema.
