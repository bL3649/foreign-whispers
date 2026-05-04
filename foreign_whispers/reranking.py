"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    char_budget = int(target_duration_s * 15)
    candidates = []

    if len(baseline_es) <= char_budget:
        candidates.append(TranslationCandidate(baseline_es, len(baseline_es), "already <= char budget"))

    without_repetition = []
    words = baseline_es.split()

    plain_words = []

    for word in words:
        new_word = ""
        for c in word:
            if c.isalnum():
                new_word += c
        plain_words.append(new_word)

    if len(words) > 0:
        without_repetition.append(words[0])
        current_word = plain_words[0]
        for i in range(1, len(words)):
            if plain_words[i].lower() != current_word.lower():
                without_repetition.append(words[i])
                current_word = plain_words[i]
    
    without_repetition = " ".join(without_repetition)
    
    if len(without_repetition) <= char_budget and without_repetition != baseline_es:
        candidates.append(TranslationCandidate(without_repetition, len(without_repetition), "removed adjacent repeated words"))
    
    connectors = {"y", "e", "o", "u"}

    without_repeated_connectors = []

    i = 0
    while i < len(words):
        without_repeated_connectors.append(words[i])
        current_word = plain_words[i].lower()

        while (
            i + 2 < len(words)
            and plain_words[i + 1].lower() in connectors
            and plain_words[i + 2].lower() == current_word
        ):
            i += 2

        i += 1
    
    without_repeated_connectors = " ".join(without_repeated_connectors)

    if len(without_repeated_connectors) <= char_budget and without_repeated_connectors != baseline_es and without_repeated_connectors != without_repetition:
        candidates.append(TranslationCandidate(without_repeated_connectors, len(without_repeated_connectors), "collapsed repeated connector phrase"))


    filler_set = {
        "a decir verdad",
        "a fin de cuentas",
        "a lo mejor",
        "a todo esto",
        "al final",
        "al final del día",
        "al parecer",
        "algo así",
        "aparentemente",
        "así que",
        "básicamente",
        "bueno",
        "bueno pues",
        "casi que",
        "ciertamente",
        "claro",
        "claramente",
        "como quien dice",
        "como un hecho fáctico",
        "cómo decirlo",
        "de alguna manera",
        "de algún modo",
        "de cierto modo",
        "de hecho",
        "de verdad",
        "digamos",
        "digamos que",
        "dicho de otra forma",
        "dicho de otro modo",
        "dicho esto",
        "dicho eso",
        "eh",
        "em",
        "en cierto modo",
        "en cierto sentido",
        "en cierta medida",
        "en el fondo",
        "en fin",
        "en gran medida",
        "en pocas palabras",
        "en realidad",
        "en resumidas cuentas",
        "en términos de",
        "en última instancia",
        "es decir",
        "esencialmente",
        "este",
        "francamente",
        "hay que decir que",
        "honestamente",
        "la cosa es que",
        "la cuestión es que",
        "la verdad",
        "la verdad es que",
        "la realidad es que",
        "literalmente",
        "lo que pasa es que",
        "lo que quiero decir es",
        "mira",
        "mire",
        "mmm",
        "no sé",
        "obviamente",
        "o sea",
        "para ser honestos",
        "para ser sincero",
        "para ser sincera",
        "por así decirlo",
        "por decirlo así",
        "por lo general",
        "por supuesto",
        "pues",
        "pues bueno",
        "pues nada",
        "realmente",
        "sabes",
        "sinceramente",
        "sin duda",
        "tipo",
        "uh",
        "um",
        "un poco",
        "verdaderamente",
        "y bueno",
        "¿me entiendes?",
        "¿me explico?",
        "¿no?",
        "¿sabes?",
        "¿sabes lo que quiero decir?",
    }

    baseline_es_copy = baseline_es

    for filler in filler_set:
        search_index = 0
        while True:
            index = baseline_es_copy.lower().find(filler.lower(), search_index) 
            condition1 = index != -1
            if not condition1:
                break
            else:
                condition2 = index == 0 or (index != 0 and not baseline_es_copy[index - 1].isalnum())
                condition3 = index == len(baseline_es_copy) - len(filler) or (index != len(baseline_es_copy) - len(filler) and not baseline_es_copy[index + len(filler)].isalnum())
                if not(condition2 and condition3):
                    search_index = index + len(filler)
                    continue
            baseline_es_copy = baseline_es_copy[:index] + baseline_es_copy[index + len(filler):]

    without_filler = ""
    space_flag = False

    for char in baseline_es_copy:
        if char == " " and space_flag == False:
            without_filler += char
            space_flag = True
        elif char != " ":
            without_filler += char
            space_flag = False
    
    without_filler = without_filler.strip()

    if len(without_filler) <= char_budget and without_filler != without_repeated_connectors and without_filler != without_repetition and without_filler != baseline_es:
        candidates.append(TranslationCandidate(without_filler, len(without_filler), "removed filler words and phrases"))
    
    common_phrases_set = {
        "en este momento": "ahora",
        "en este instante": "ahora",
        "en estos momentos": "ahora",
        "en la actualidad": "hoy",
        "actualmente": "hoy",
        "hoy en día": "hoy",
        "en el día de hoy": "hoy",
        "a día de hoy": "hoy",
        "en este punto": "ahora",
        "en ese momento": "entonces",
        "en aquel momento": "entonces",
        "en ese instante": "entonces",
        "en el momento en que": "cuando",
        "en el momento en el que": "cuando",
        "al momento de": "al",
        "tan pronto como": "cuando",
        "una vez que": "cuando",
        "cada vez que": "cuando",
        "durante el tiempo que": "mientras",
        "al mismo tiempo": "a la vez",
        "de manera simultánea": "a la vez",
        "simultáneamente": "a la vez",
        "a lo largo de": "durante",
        "a lo largo del tiempo": "con el tiempo",
        "con el paso del tiempo": "con el tiempo",
        "posteriormente": "después",
        "anteriormente": "antes",
        "previamente": "antes",
        "de forma inmediata": "de inmediato",
        "inmediatamente": "ya",
        "de vez en cuando": "a veces",
        "en ocasiones": "a veces",
        "en algunas ocasiones": "a veces",
        "ocasionalmente": "a veces",
        "en numerosas ocasiones": "muchas veces",
        "con frecuencia": "seguido",
        "frecuentemente": "seguido",
        "muy a menudo": "seguido",
        "la mayoría de las veces": "casi siempre",
        "en la mayoría de los casos": "casi siempre",
        "por lo general": "normalmente",
        "generalmente": "normalmente",
        "habitualmente": "normalmente",
        "de manera constante": "siempre",
        "constantemente": "siempre",
        "con respecto a": "sobre",
        "respecto a": "sobre",
        "respecto de": "sobre",
        "en cuanto a": "sobre",
        "en lo que respecta a": "sobre",
        "en relación con": "sobre",
        "en referencia a": "sobre",
        "referente a": "sobre",
        "acerca de": "sobre",
        "en torno a": "sobre",
        "en materia de": "sobre",
        "en el ámbito de": "en",
        "en el campo de": "en",
        "en el terreno de": "en",
        "en términos de": "sobre",
        "desde el punto de vista de": "desde",
        "bajo el punto de vista de": "desde",
        "debido al hecho de que": "porque",
        "debido a que": "porque",
        "a causa de que": "porque",
        "por causa de": "por",
        "como consecuencia de": "por",
        "como resultado de": "por",
        "a raíz de": "por",
        "en vista de que": "porque",
        "dado que": "porque",
        "ya que": "porque",
        "puesto que": "porque",
        "por el hecho de que": "porque",
        "gracias a que": "porque",
        "por motivo de": "por",
        "por razones de": "por",
        "por lo tanto": "así",
        "por tanto": "así",
        "por consiguiente": "así",
        "en consecuencia": "así",
        "como consecuencia": "así",
        "como resultado": "así",
        "de ahí que": "por eso",
        "por esa razón": "por eso",
        "por ese motivo": "por eso",
        "por esta razón": "por eso",
        "por este motivo": "por eso",
        "esto significa que": "significa que",
        "eso significa que": "significa que",
        "lo que significa que": "o sea que",
        "con el fin de": "para",
        "a fin de": "para",
        "con el objetivo de": "para",
        "con el propósito de": "para",
        "con la intención de": "para",
        "con la finalidad de": "para",
        "con miras a": "para",
        "de cara a": "para",
        "para poder": "para",
        "a efectos de": "para",
        "en el caso de que": "si",
        "en caso de que": "si",
        "siempre y cuando": "si",
        "siempre que": "si",
        "a condición de que": "si",
        "con tal de que": "si",
        "a pesar de que": "aunque",
        "a pesar del hecho de que": "aunque",
        "pese a que": "aunque",
        "sin embargo": "pero",
        "no obstante": "pero",
        "a pesar de todo": "igual",
        "con todo": "igual",
        "de todos modos": "igual",
        "de todas formas": "igual",
        "de todas maneras": "igual",
        "en cualquier caso": "igual",
        "sea como sea": "igual",
        "por otra parte": "además",
        "por otro lado": "además",
        "además de eso": "además",
        "adicionalmente": "además",
        "asimismo": "también",
        "de igual manera": "igual",
        "de la misma manera": "igual",
        "del mismo modo": "igual",
        "igualmente": "igual",
        "en definitiva": "en suma",
        "en conclusión": "en suma",
        "en resumen": "en suma",
        "resumiendo": "en suma",
        "en pocas palabras": "en suma",
        "en resumidas cuentas": "en suma",
        "a fin de cuentas": "al final",
        "en última instancia": "al final",
        "al fin y al cabo": "al final",
        "al final del día": "al final",
        "en primer lugar": "primero",
        "en segundo lugar": "segundo",
        "en tercer lugar": "tercero",
        "para empezar": "primero",
        "para terminar": "finalmente",
        "por último": "finalmente",
        "quiero decir que": "o sea",
        "lo que quiero decir es": "o sea",
        "es decir": "o sea",
        "esto es": "o sea",
        "dicho de otra manera": "o sea",
        "dicho de otra forma": "o sea",
        "dicho de otro modo": "o sea",
        "en otras palabras": "o sea",
        "por así decirlo": "digamos",
        "por decirlo así": "digamos",
        "en esencia": "básicamente",
        "fundamentalmente": "básicamente",
        "la realidad es que": "en realidad",
        "la verdad es que": "en verdad",
        "lo cierto es que": "en verdad",
        "hay que decir que": "cabe decir",
        "cabe señalar que": "cabe decir",
        "cabe destacar que": "cabe decir",
        "vale la pena señalar que": "cabe decir",
        "es importante señalar que": "cabe decir",
        "es importante destacar que": "cabe decir",
        "más o menos": "aprox.",
        "aproximadamente": "aprox.",
        "alrededor de": "unos",
        "cerca de": "unos",
        "en torno a": "unos",
        "del orden de": "unos",
        "una especie de": "un",
        "un tipo de": "un",
        "una clase de": "un",
        "un gran número de": "muchos",
        "una gran cantidad de": "mucho",
        "una cantidad significativa de": "mucho",
        "cantidad significativa de": "mucho",
        "un número considerable de": "muchos",
        "una cantidad considerable de": "mucho",
        "la mayor parte de": "la mayoría",
        "gran parte de": "mucho",
        "buena parte de": "mucho",
        "una pequeña cantidad de": "poco",
        "un pequeño número de": "pocos",
        "en gran medida": "mucho",
        "en cierta medida": "algo",
        "en cierto grado": "algo",
        "hasta cierto punto": "algo",
        "relativamente": "algo",
        "considerablemente": "mucho",
        "significativamente": "mucho",
        "extremadamente": "muy",
        "sumamente": "muy",
        "enormemente": "mucho",
        "muchísimo": "mucho",
        "tiene la capacidad de": "puede",
        "tienen la capacidad de": "pueden",
        "tenía la capacidad de": "podía",
        "tenían la capacidad de": "podían",
        "es capaz de": "puede",
        "son capaces de": "pueden",
        "era capaz de": "podía",
        "eran capaces de": "podían",
        "puede llegar a": "puede",
        "pueden llegar a": "pueden",
        "se ve obligado a": "debe",
        "se ven obligados a": "deben",
        "está obligado a": "debe",
        "están obligados a": "deben",
        "tiene que": "debe",
        "tienen que": "deben",
        "debe de": "debe",
        "deben de": "deben",
        "necesita": "debe",
        "necesitan": "deben",
        "es necesario que": "debe",
        "hace falta que": "debe",
        "se requiere que": "debe",
        "hace referencia a": "habla de",
        "hacen referencia a": "hablan de",
        "se refiere a": "habla de",
        "se refieren a": "hablan de",
        "tiene que ver con": "trata de",
        "tienen que ver con": "tratan de",
        "está relacionado con": "trata de",
        "están relacionados con": "tratan de",
        "guarda relación con": "trata de",
        "tiene relación con": "trata de",
        "trata acerca de": "trata de",
        "habla acerca de": "habla de",
        "menciona el hecho de que": "dice que",
        "afirma que": "dice que",
        "señala que": "dice que",
        "indica que": "dice que",
        "manifiesta que": "dice que",
        "expresa que": "dice que",
        "sostiene que": "dice que",
        "declara que": "dice que",
        "asegura que": "dice que",
        "explica que": "dice que",
        "da a entender que": "sugiere que",
        "pone de manifiesto que": "muestra que",
        "deja claro que": "muestra que",
        "dar a conocer": "mostrar",
        "darse cuenta de": "notar",
        "tomar conciencia de": "notar",
        "se lleva a cabo": "ocurre",
        "se llevan a cabo": "ocurren",
        "tiene lugar": "ocurre",
        "tienen lugar": "ocurren",
        "se produce": "ocurre",
        "se producen": "ocurren",
        "se da": "ocurre",
        "se dan": "ocurren",
        "ocasiona": "causa",
        "provoca": "causa",
        "genera": "causa",
        "da lugar a": "causa",
        "conduce a": "causa",
        "lleva a": "causa",
        "resulta en": "causa",
        "termina en": "acaba en",
        "comienza a": "empieza a",
        "da comienzo a": "empieza",
        "pone fin a": "acaba",
        "llega a su fin": "acaba",
        "continúa siendo": "sigue siendo",
        "se empieza a ver": "se ve",
        "llevar a cabo": "hacer",
        "lleva a cabo": "hace",
        "llevan a cabo": "hacen",
        "llevaron a cabo": "hicieron",
        "hacer uso de": "usar",
        "hace uso de": "usa",
        "hacen uso de": "usan",
        "utilizar": "usar",
        "utiliza": "usa",
        "utilizan": "usan",
        "emplear": "usar",
        "emplea": "usa",
        "emplean": "usan",
        "poner en marcha": "iniciar",
        "pone en marcha": "inicia",
        "ponen en marcha": "inician",
        "dar inicio a": "iniciar",
        "dar comienzo a": "iniciar",
        "poner en práctica": "aplicar",
        "pone en práctica": "aplica",
        "ponen en práctica": "aplican",
        "tomar en consideración": "considerar",
        "tener en cuenta": "considerar",
        "tiene en cuenta": "considera",
        "tienen en cuenta": "consideran",
        "tomar en cuenta": "considerar",
        "hacer frente a": "enfrentar",
        "hace frente a": "enfrenta",
        "hacen frente a": "enfrentan",
        "enfrentarse a": "enfrentar",
        "sacar provecho de": "aprovechar",
        "saca provecho de": "aprovecha",
        "aprovecharse de": "aprovechar",
        "prestar atención a": "atender",
        "presta atención a": "atiende",
        "poner atención a": "atender",
        "dar respuesta a": "responder",
        "da respuesta a": "responde",
        "dar apoyo a": "apoyar",
        "brindar apoyo a": "apoyar",
        "prestar ayuda a": "ayudar",
        "brindar ayuda a": "ayudar",
        "dar ayuda a": "ayudar",
        "tomar una decisión": "decidir",
        "toma una decisión": "decide",
        "tomaron una decisión": "decidieron",
        "hacer una pregunta": "preguntar",
        "hace una pregunta": "pregunta",
        "hacer una llamada": "llamar",
        "hace una llamada": "llama",
        "hacer una visita": "visitar",
        "hace una visita": "visita",
        "dar un paseo": "pasear",
        "echar un vistazo": "mirar",
        "dar un vistazo": "mirar",
        "poner en duda": "dudar de",
        "poner en riesgo": "arriesgar",
        "estar en riesgo": "peligrar",
        "estar de acuerdo": "coincidir",
        "ponerse de acuerdo": "acordar",
        "llegar a un acuerdo": "acordar",
        "llegar a la conclusión": "concluir",
        "llega a la conclusión": "concluye",
        "llegan a la conclusión": "concluyen",
        "tener la intención de": "querer",
        "tiene la intención de": "quiere",
        "tienen la intención de": "quieren",
        "tener la oportunidad de": "poder",
        "tiene la oportunidad de": "puede",
        "tienen la oportunidad de": "pueden",
        "impacto real": "impacto",
        "impacto significativo": "gran impacto",
        "cantidad significativa": "gran cantidad",
        "número significativo": "gran número",
        "crisis aguda": "crisis grave",
        "período prolongado": "mucho tiempo",
        "cierre prolongado": "cierre largo",
        "supuesto fundamental": "base",
        "mal día": "día malo",
        "muy malo": "pésimo",
        "muy grande": "enorme",
        "muy pequeño": "mínimo",
        "muy importante": "clave",
        "muy difícil": "duro",
        "muy fácil": "simple",
        "muy rápido": "rápido",
        "muy lento": "lento",
        "gran ganador": "ganador",
        "ganadora relativa": "beneficiada",
        "de manera encubierta": "en secreto",
        "sin ser detectados": "en secreto",
        "de forma secreta": "en secreto",
        "de manera clara": "claramente",
        "de forma clara": "claramente",
        "de manera directa": "directamente",
        "de forma directa": "directamente",
        "de manera rápida": "rápidamente",
        "de forma rápida": "rápidamente",
        "de manera lenta": "lentamente",
        "de forma lenta": "lentamente",
        "de manera fácil": "fácilmente",
        "de forma fácil": "fácilmente",
        "de manera difícil": "difícilmente",
        "de forma oficial": "oficialmente",
        "de manera oficial": "oficialmente",
        "de forma temporal": "temporalmente",
        "de manera temporal": "temporalmente",
        "de forma permanente": "permanentemente",
        "de manera permanente": "permanentemente",
    }

    baseline_es_copy = baseline_es

    for phrase, replacement in sorted(common_phrases_set.items(), key=lambda x: len(x[0]), reverse=True):
        search_index = 0
        while True:
            index = baseline_es_copy.lower().find(phrase.lower(), search_index)
            condition1 = index != -1
            if not condition1:
                break
            else:
                condition2 = index == 0 or (index != 0 and not baseline_es_copy[index - 1].isalnum())
                condition3 = index == len(baseline_es_copy) - len(phrase) or (index != len(baseline_es_copy) - len(phrase) and not baseline_es_copy[index + len(phrase)].isalnum())
                if not(condition2 and condition3):
                    search_index = index + len(phrase)
                    continue
            baseline_es_copy = baseline_es_copy[:index] + replacement + baseline_es_copy[index + len(phrase):]

    common_phrases_contracted = baseline_es_copy

    if len(common_phrases_contracted) <= char_budget and common_phrases_contracted != without_filler and common_phrases_contracted != without_repeated_connectors and common_phrases_contracted != without_repetition and common_phrases_contracted != baseline_es:
        candidates.append(TranslationCandidate(common_phrases_contracted, len(common_phrases_contracted), "contracted common phrases"))

    if candidates == []:
        smarter_truncation = ""
        char_count = 0

        for word in baseline_es.split():
            if char_count + len(word) <= char_budget:
                smarter_truncation += word
                char_count += len(word)
            else:
                break

            if char_count + 1 <= char_budget:
                smarter_truncation += " "
                char_count += 1
            else:
                break

        smarter_truncation = smarter_truncation.strip()
        if smarter_truncation != "":
            candidates.append(TranslationCandidate(smarter_truncation, len(smarter_truncation), "word-boundary truncation"))

    logger.info(
        "get_shorter_translations called for %.1fs budget (%d chars baseline) — "
        "returning translation candidates.",
        target_duration_s,
        len(baseline_es),
    )
    return sorted(candidates, key=lambda x: x.char_count)
