FOLK = {"Federico", "Pranav", "Riccardo"}
INITIAL_TEXT = "Ciao, come posso esserti d'aiuto oggi?"
SEARCH_TEXT = "Ecco qualche immagine per te"


def different_person(p_new, p_old):
    return "Ciao {}, cosa hai fatto a {}? Devo preoccuparmi?" \
        .format(p_new, p_old)


def answare(ans, p):
    if p not in FOLK:
        return "Non prendo ordini da te!"
    out = {
        "sì": "Ciao {}, mi fa piacere esserti d'aiuto".format(p),
        "no": "Mi spiace {}, starò più attento".format(p),
        "forse": "Senza offesa {}, potresti essere più preciso?".format(p),
        "non capico": "Potresti parlare più chiaramente per favore?"
    }
    return out[ans]


def text_generator():
    old = ""

    def generate_text(ans, p):
        nonlocal old
        if p and old and old != p:
            old_p, old = old, p
            return different_person(p, old_p)
        old = p
        return answare(ans, p)

    return generate_text
