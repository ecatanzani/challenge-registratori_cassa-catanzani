# Valutazione delle prestazioni

Questa pagina riporta le metriche di riferimento del sistema e i tempi di
risposta, misurati sul codice attuale con `scripts/eval.py`.

## Come si riproduce

```bash
python scripts/eval.py
```

Lo script interroga il sistema completo, LLM compreso, su tutte le domande dell'eval set e confronta le risposte con le etichette. Ogni domanda passa da `answer_query()` e lascia una riga nell'audit log. Il file di audit viene creato alla prima esecuzione del codice (se non esiste) e poi aggiornato incrementalmente. Questo file riporta al suo interno tutte le informazioni in merito alle prestazioni del sistema, comprensivo della query dell'utente.

Le medie e i percentili dei tempi riportati qui sotto sono calcolati proprio da quell'audit log, che per ogni domanda registra tempo di retrieval, tempo al primo token e tempo totale.

## Condizioni della misura

| | |
|---|---|
| eval set | `eval/eval_set.jsonl`, 40 domande etichettate |
| manuale indicizzato | `printf_f` v03, 73 pagine, 196 chunk, 17 immagini |
| modello di risposta | `claude-sonnet-5` |
| embedding | `intfloat/multilingual-e5-large` |
| reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| candidati | 8 densi, 8 BM25, 5 dopo la fusione |
| soglia del gate di dominio | 0.8 |

L'eval set ha tre classi:

| classe | domande | comportamento atteso |
|---|---|---|
| `rispondibile` | 24 (di cui 2 con refusi) | risposta con i passi della procedura |
| `vicina_non_documentata` | 10 | argomento del negozio ma assente dal manuale: fallback con FAQ |
| `fuori_tema` | 6 | estranea al dominio: rifiuto |

## Metriche di riferimento

### Retrieval

Le prestazioni del retrieval sono state calcolate sulle 24 domande con risposta: viene valutato se, fra i frammenti recuperati dopo il reranker, ce n'è uno di una pagina attesa.

| metrica | valore | cosa misura |
|---|---|---|
| hit@1 | 19/24 (79%) | il primo frammento è di una pagina giusta |
| hit@3 | 23/24 (96%) | ce n'è uno giusto fra i primi tre |
| **hit@5** | **24/24 (100%)** | ce n'è uno giusto fra i cinque passati all'LLM |
| MRR | 0.872 | media di 1/posizione del primo frammento giusto |

Hit@5 è la metrica più importante: sono i cinque frammenti che l'LLM riceve; se la pagina giusta è fra quelli il modello ha il materiale per rispondere.

MRR sta per *Mean Reciprocal Rank*; è una metrica usata nei sistemi di Information Retrieval e RAG per valutare la capacità del sistema di posizionare il primo risultato pertinente il più in alto possibile nella classifica.

### Decisioni: rispondere, fare fallback, rifiutare

Il sistema può decidere di comportarsi in uno dei seguenti modi:

- Rispondere: se la query passa il gate di dominio
- Fallback: se la query passa il gate di dominio ma la query non è aderente al manuale
- Rifiutare di rispondere: se la query non passa il gate di dominio


| classe | decisioni corrette |
|---|---|
| `rispondibile` | 24/24 |
| `vicina_non_documentata` | 10/10 |
| `fuori_tema` | 6/6 (5 fermate dal gate, 1 fallback dell'LLM) |
| **complessivo** | **40/40 (100%)** |

Nei test eseguiti non sono presenti errori. La domanda meno stabile è «Come aggiungo un nuovo operatore al gestionale di magazzino?»: può ricevere la procedura di programmazione degli operatori del registratore (p.21) invece del fallback. In passato succedeva in tutte le esecuzioni; nelle 10 esecuzioni dell'eval del 14 settembre sul codice attuale è successo una volta, sempre con gli stessi frammenti recuperati, quindi a cambiare è la decisione dell'LLM. La causa è un'omonimia: il manuale documenta davvero l'aggiunta di un operatore, ma della cassa, non del gestionale citato nella domanda. Né il gate di dominio né la validazione del grounding possono intercettarlo, perché la domanda è vicina al dominio e ogni passo cita un frammento realmente recuperato: il grounding verifica la provenienza della risposta, non che riguardi l'oggetto chiesto.

Una possibile soluzione potrebbe essere la seguente:

Aggiungere a `StructuredAnswer` un campo che obblighi il modello a pronunciarsi sulla copertura, per esempio stesso_oggetto: bool, con una descrizione come "True se i frammenti descrivono proprio il sistema nominato nella domanda; False se descrivono solo qualcosa di omonimo, come gli operatori della cassa invece che del gestionale". Se è False, `chain.py` trasforma la risposta in fallback, con un chiarimento mirato.

### Immagini

| metrica | valore |
|---|---|
| figure attese allegate alla risposta | 12/12 (100%) |

Una figura conta se viene allegata da una delle pagine attese, fino al numero di
figure che la domanda richiede.

## Tempi di risposta

I budget indicati sono 2 s per il retrieval e 8 s per la risposta
completa.

### A regime

Esclusa la prima domanda dell'esecuzione:

| fase | media | mediana | p90 | massimo |
|---|---|---|---|---|
| retrieval | **194 ms** | 210 ms | 237 ms | 257 ms |
| primo token dell'LLM, dalla domanda | **1315 ms** | 1284 ms | 1628 ms | 1872 ms |
| risposta completa | **4870 ms** | 4456 ms | 8447 ms | 10134 ms |

Il tempo al primo token è misurato dall'arrivo della domanda, quindi comprende anche il retrieval; è calcolato sulle 34 domande che arrivano all'LLM. Nessun retrieval supera il budget di 2 s; 6 risposte complete su 39 superano gli 8 s, tutte procedure con passi (massimo 10,1 s).

### Per tipo di risposta

| percorso | domande | media | mediana | massimo |
|---|---|---|---|---|
| risposta con i passi della procedura | 23 | 6252 ms | 5757 ms | 10134 ms |
| fallback deciso dall'LLM | 11 | 4156 ms | 4378 ms | 4687 ms |
| fermata dal gate, senza LLM | 5 | 79 ms | 48 ms | 138 ms |

La risposta completa dipende soprattutto dalla generazione: una procedura con passi, citazioni e immagini richiede circa 2 secondi in più di un fallback. Le domande fermate dal gate costano meno di un decimo di secondo.

### Variabilità

I tempi dell'LLM cambiano da un'esecuzione all'altra. Sulle sei esecuzioni complete dell'eval fatte durante lo sviluppo, con lo stesso eval set e versioni che differivano solo nella correzione dei refusi, la mediana della risposta completa è andata da 4435 a 5087 ms.