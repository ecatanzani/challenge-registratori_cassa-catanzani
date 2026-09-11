"""Riporto della domanda in sospeso quando il sistema ha chiesto un chiarimento.
"""

# Oltre questa lunghezza l'input e' una domanda a se' stante, non la risposta
# a un chiarimento: chi risponde "E43" scrive due caratteri, chi cambia
# argomento scrive una frase.
MAX_PAROLE_RISPOSTA = 6

INTERROGATIVI = frozenset(
    "come cosa quando dove quale quali perche perché chi quanto quanti quanta "
    "quante posso devo dove'".split()
)


def sembra_risposta_a_chiarimento(testo: str) -> bool:
    """True se l'input va letto come risposta al chiarimento appena chiesto.

    I tre segnali sono negativi:
    - la presenza di un punto interrogativo
    - una lunghezza da frase compiuta
    - un incipit interrogativo.
    
    In assenza di tutti e tre, e' una risposta.
    """
    parole = testo.strip().split()
    if not parole or "?" in testo:
        return False
    if len(parole) > MAX_PAROLE_RISPOSTA:
        return False
    return parole[0].lower().strip(",;:") not in INTERROGATIVI


def componi_query(domanda_in_sospeso: str | None, testo: str) -> str:
    """Query da inviare a `answer_query()`. Valuta se ci sono chiarimenti
    ed eventualmente crea una query composita
    """
    testo = testo.strip()
    if domanda_in_sospeso and sembra_risposta_a_chiarimento(testo):
        return f"{domanda_in_sospeso.rstrip(' ?')} — {testo}"
    return testo
