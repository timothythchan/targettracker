"""
SpacyTargetExtractor — Layer 1 Baseline NLP Extraction
=======================================================
Replicates the Cohen & Nguyen (2024) "Moving Targets" NLP extraction
methodology using spaCy NER and dependency parsing (pp. 8-10).

The paper defines a "performance target" by walking from any Named Entity
with label in {MONEY, PERCENT, PRODUCT} to the *qualitative* noun being
targeted:

  * PRODUCT   -> emit the entity's enclosing noun chunk
                 (e.g. ``Mac`` -> ``Mac``).
  * MONEY     -> walk attr -> AUX/copula -> nsubj and emit the subject
                 noun chunk (e.g. ``$1.67 billion`` in
                 ``Net income was $1.67 billion`` -> ``Net income``).
  * PERCENT   -> walk to the modified head NOUN, then prep -> pobj, and
                 emit the prep object's noun chunk (e.g. ``12%`` in
                 ``12% year-over-year increase in Mac sales`` ->
                 ``Mac sales``).  Falls back to the head noun if no
                 prep/pobj is present.

The legacy ``mode="strict"`` extractor (noun-chunk + numeric-modifier +
goal-verb gate) is retained behind a flag for ablation studies.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

logger = logging.getLogger("earningslens.baseline.extractor")

# ---------------------------------------------------------------------------
# Constants (mirror config.yaml baseline section)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paper-faithful constants. Cohen & Nguyen (2024) "Moving Targets", pp. 8-10.
# ---------------------------------------------------------------------------

# Paper, p. 9: "named entities that are Products, Money, or Percent."
# CARDINAL and QUANTITY are intentionally excluded (they pull in raw
# numbers and physical quantities that aren't performance targets).
NER_LABELS: frozenset[str] = frozenset(["MONEY", "PERCENT", "PRODUCT"])

FINANCIAL_LABELS: frozenset[str] = frozenset(["MONEY", "PERCENT"])

# Goal-verb list is kept for backwards compatibility with `mode="strict"`
# but is NOT used in the paper-faithful default mode.
GOAL_VERBS: frozenset[str] = frozenset(
    [
        "achieve", "target", "maintain", "expect", "deliver", "reach",
        "sustain", "exceed", "grow", "generate", "drive", "hit", "attain",
        "improve", "increase",
    ]
)

NUMERIC_POS: frozenset[str] = frozenset(["NUM"])
NUMERIC_DEPS: frozenset[str] = frozenset(["nummod", "quantmod", "num"])

# Copular / linking auxiliaries used in MONEY "attr -> AUX -> nsubj" walks.
# The paper's worked example: "Net income was $1.67 billion" -> nsubj of
# `was` is `Net income`.
_COPULA_LEMMAS: frozenset[str] = frozenset(
    ["be", "become", "remain", "total", "reach", "hit", "equal", "come"]
)


# ---------------------------------------------------------------------------
# SpacyTargetExtractor
# ---------------------------------------------------------------------------


class SpacyTargetExtractor:
    """
    Extract performance targets from earnings call transcript segments
    following the Cohen & Nguyen (2024) Moving Targets methodology.

    Two extraction modes are supported:

      * ``mode="paper"`` (default) -- faithful Cohen & Nguyen (2024)
        replication. PRODUCT entities emit their enclosing noun chunk;
        MONEY entities walk ``attr -> AUX/copula -> nsubj`` and emit the
        subject noun (e.g. ``$1.67 billion`` -> ``Net income``); PERCENT
        entities walk ``nmod -> noun -> prep -> pobj`` and emit the
        prepositional object (e.g. ``12% increase in Mac sales`` ->
        ``Mac sales``). No goal-verb filter.

      * ``mode="strict"`` -- legacy behaviour: emits the noun chunk that
        contains the entity, gated by a goal-verb whitelist. Useful as a
        precision-biased ablation alongside the paper baseline.

    Parameters
    ----------
    model_name : str
        spaCy model to load.  Defaults to ``en_core_web_sm`` (the model
        used in the paper). Pass ``en_core_web_lg`` for higher-quality
        NER on long-tail product names.
    mode : str
        ``"paper"`` or ``"strict"``. Default ``"paper"``.
    goal_verbs : set[str], optional
        Override the goal-verb list (only used when ``mode="strict"``).
    ner_labels : set[str], optional
        Override the canonical NER label set.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        mode: str = "paper",
        goal_verbs: Optional[frozenset[str]] = None,
        ner_labels: Optional[frozenset[str]] = None,
    ) -> None:
        if mode not in {"paper", "strict"}:
            raise ValueError(f"mode must be 'paper' or 'strict', got {mode!r}")
        self.model_name = model_name
        self.mode = mode
        self.goal_verbs: frozenset[str] = goal_verbs or GOAL_VERBS
        self.ner_labels: frozenset[str] = ner_labels or NER_LABELS

        logger.info("Loading spaCy model: %s", model_name)
        try:
            self.nlp: Language = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                "Run: python -m spacy download en_core_web_lg"
            ) from exc
        logger.info("spaCy model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_targets(
        self, text: str, component_type: int
    ) -> List[Dict]:
        """
        Extract performance targets from a single transcript text segment.

        Parameters
        ----------
        text : str
            Raw transcript text for one component (e.g. management presentation).
        component_type : int
            Numeric code for the call component:
            2 = Presentation, 3 = Analyst Question, 4 = Management Answer.

        Returns
        -------
        List[Dict]
            Each dict has keys: target_text, entity_label, numeric_value,
            governing_verb, is_financial, sentence, component_type.
        """
        if not text or not text.strip():
            return []

        doc: Doc = self.nlp(text)
        results: List[Dict] = []

        for sent in doc.sents:
            sent_targets = self._extract_from_sentence(sent, component_type)
            results.extend(sent_targets)

        logger.debug(
            "extract_targets: component_type=%d, found %d targets",
            component_type,
            len(results),
        )
        return results

    def extract_from_transcript(
        self, transcript_components: List[Dict]
    ) -> List[Dict]:
        """
        Process all components of one earnings call transcript.

        Parameters
        ----------
        transcript_components : List[Dict]
            Each dict must have at least:
              - ``text`` (str): The raw text of the component.
              - ``component_type`` (int): 2, 3, or 4.

        Returns
        -------
        List[Dict]
            All extracted targets across all components.
        """
        all_targets: List[Dict] = []
        for component in transcript_components:
            text = component.get("text", "")
            ctype = component.get("component_type", 0)
            try:
                targets = self.extract_targets(text, ctype)
                all_targets.extend(targets)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to extract from component (type=%s): %s",
                    ctype,
                    exc,
                )
        return all_targets

    def normalize_target(self, target_text: str) -> str:
        """
        Normalize a target phrase for cross-quarter matching.

        The goal is to produce a string that captures the *qualitative*
        target (e.g. "revenue growth", "operating margin") while removing
        the *quantitative* magnitude (e.g. "15%", "$2 billion") and
        boilerplate hedges ("approximately", "about"). Two quarters of
        guidance are considered the same target if their normalized forms
        match, even when the magnitude has been revised.

        Steps:
          1. Lowercase.
          2. Strip currency symbols ($, €, £, ¥).
          3. Strip raw numbers (integers, decimals, comma-separated).
          4. Strip magnitude / unit tokens (%, percent, bps, basis points,
             million, billion, trillion, mn, bn, tn, x).
          5. Strip hedging modifiers (approximately, about, around,
             roughly, nearly, over, up to, at least, at most, between,
             range of, low/mid/high single/double/triple digits).
          6. Lemmatize every remaining token via the spaCy model.
          7. Drop leading determiners / possessives (the, a, an, our…).
          8. Drop punctuation / whitespace tokens.
          9. Collapse runs of whitespace.

        Parameters
        ----------
        target_text : str
            Raw noun-phrase text.

        Returns
        -------
        str
            Normalized string suitable for set-based matching.
        """
        if not target_text:
            return ""

        s = target_text.lower().strip()

        # 2. Currency symbols
        s = re.sub(r"[\$€£¥]", " ", s)

        # 3. Numbers (1, 1.5, 1,000, 1,234.56). Run before unit stripping
        #    so trailing units detach cleanly.
        s = re.sub(r"\b\d+(?:[.,]\d+)*\b", " ", s)

        # 4. Units / magnitudes
        s = re.sub(
            r"\b(percent(?:age)?|pct|bps|basis\s+points?|bp|"
            r"million|billion|trillion|thousand|"
            r"mn|bn|tn|mm|k|x)\b",
            " ",
            s,
        )
        s = re.sub(r"%", " ", s)

        # 5. Hedging / approximation modifiers
        s = re.sub(
            r"\b(approximately|approx\.?|about|around|roughly|nearly|"
            r"almost|over|under|above|below|more\s+than|less\s+than|"
            r"up\s+to|at\s+least|at\s+most|no\s+(?:more|less)\s+than|"
            r"between|range\s+of|in\s+the\s+range\s+of|north\s+of|"
            r"south\s+of|low|mid|high|single|double|triple|digits?|"
            r"plus|minus)\b",
            " ",
            s,
        )

        # Collapse whitespace before passing to spaCy so token boundaries
        # are clean.
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return ""

        # 6-8. Lemmatize and drop determiners / punctuation
        doc = self.nlp(s)
        lemmas: List[str] = []
        for i, tok in enumerate(doc):
            if tok.is_punct or tok.is_space:
                continue
            # Drop leading determiners / possessives
            if (
                not lemmas
                and tok.pos_ in {"DET", "PRON"}
                and tok.dep_ in {"det", "poss", "nsubj"}
            ):
                continue
            lemmas.append(tok.lemma_)

        normalized = " ".join(lemmas).strip()
        # 9. Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_from_sentence(
        self, sent: Span, component_type: int
    ) -> List[Dict]:
        """Extract performance targets from a single spaCy Sentence Span.

        Dispatches to the paper-faithful or strict extractor based on
        ``self.mode``.
        """
        if self.mode == "paper":
            return self._extract_paper_mode(sent, component_type)
        return self._extract_strict_mode(sent, component_type)

    # ------------------------------------------------------------------
    # Paper-faithful extractor (Cohen & Nguyen 2024, pp. 8-10)
    # ------------------------------------------------------------------

    def _extract_paper_mode(
        self, sent: Span, component_type: int
    ) -> List[Dict]:
        """Iterate ``sent.ents`` and dispatch by entity label.

        For every entity whose label is in ``self.ner_labels``:
          * PRODUCT  -> :meth:`_extract_product_target`
          * MONEY    -> :meth:`_extract_money_target`
          * PERCENT  -> :meth:`_extract_percent_target`

        Each helper returns a target dict (or ``None`` if extraction
        failed). Duplicate ``target_text`` values within the same sentence
        are collapsed -- the paper records each surface form once per
        sentence.
        """
        targets: List[Dict] = []
        seen_in_sent: set[str] = set()

        for ent in sent.ents:
            if ent.label_ not in self.ner_labels:
                continue

            target: Optional[Dict]
            if ent.label_ == "PRODUCT":
                target = self._extract_product_target(ent, sent, component_type)
            elif ent.label_ == "MONEY":
                target = self._extract_money_target(ent, sent, component_type)
            elif ent.label_ == "PERCENT":
                target = self._extract_percent_target(ent, sent, component_type)
            else:  # pragma: no cover -- guarded by ner_labels
                target = None

            if target is None:
                continue

            key = target["target_text"].strip().lower()
            if not key or key in seen_in_sent:
                continue
            seen_in_sent.add(key)
            targets.append(target)

        return targets

    @staticmethod
    def _enclosing_noun_chunk(token: Token) -> Optional[Span]:
        """Return the noun chunk in ``token.doc`` that contains *token*."""
        for chunk in token.doc.noun_chunks:
            if chunk.start <= token.i < chunk.end:
                return chunk
        return None

    def _extract_product_target(
        self, ent: Span, sent: Span, component_type: int
    ) -> Optional[Dict]:
        """Paper, p. 9: "All noun-chunks that are Product entity are
        recorded as a target." Emit the enclosing noun chunk text; fall
        back to the entity span itself.
        """
        chunk = self._enclosing_noun_chunk(ent.root)
        text = chunk.text if chunk is not None else ent.text
        return {
            "target_text": text.strip(),
            "entity_label": "PRODUCT",
            "numeric_value": ent.text,
            "governing_verb": "",
            "is_financial": False,
            "sentence": sent.text.strip(),
            "component_type": component_type,
        }

    def _extract_money_target(
        self, ent: Span, sent: Span, component_type: int
    ) -> Optional[Dict]:
        """Paper worked example: "Net income was $1.67 billion."

        ``$1.67 billion`` is *attr* of ``was`` (AUX, lemma ``be``).
        ``was``'s *nsubj* child is ``Net income`` -- the recorded target.

        We accept any token of the entity that has dep ``attr`` (or
        ``acomp``/``oprd`` as parser variants) whose head's lemma is in
        ``_COPULA_LEMMAS``. Falls back to the nearest ancestor noun chunk
        if no copula structure is found.
        """
        copula_deps = {"attr", "acomp", "oprd"}
        head_for_subject: Optional[Token] = None

        for tok in ent:
            if tok.dep_ in copula_deps:
                head = tok.head
                if head.pos_ in {"AUX", "VERB"} and head.lemma_.lower() in _COPULA_LEMMAS:
                    head_for_subject = head
                    break

        # Some parses attach the money to the copula via a different label.
        if head_for_subject is None:
            for tok in ent:
                head = tok.head
                if (
                    head.pos_ in {"AUX", "VERB"}
                    and head.lemma_.lower() in _COPULA_LEMMAS
                ):
                    head_for_subject = head
                    break

        target_text: Optional[str] = None
        verb_lemma: str = ""
        if head_for_subject is not None:
            verb_lemma = head_for_subject.lemma_.lower()
            for child in head_for_subject.children:
                if child.dep_ == "nsubj" or child.dep_ == "nsubjpass":
                    chunk = self._enclosing_noun_chunk(child)
                    target_text = (chunk.text if chunk is not None else child.text).strip()
                    break

        # Fallback: walk to the nearest ancestor NOUN/PROPN.
        if target_text is None:
            anc: Token = ent.root.head
            for _ in range(8):  # bounded walk
                if anc.pos_ in {"NOUN", "PROPN"}:
                    chunk = self._enclosing_noun_chunk(anc)
                    target_text = (chunk.text if chunk is not None else anc.text).strip()
                    break
                if anc.head.i == anc.i:
                    break
                anc = anc.head

        if not target_text:
            return None

        return {
            "target_text": target_text,
            "entity_label": "MONEY",
            "numeric_value": ent.text,
            "governing_verb": verb_lemma,
            "is_financial": True,
            "sentence": sent.text.strip(),
            "component_type": component_type,
        }

    def _extract_percent_target(
        self, ent: Span, sent: Span, component_type: int
    ) -> Optional[Dict]:
        """Paper worked example: "12% year-over-year increase in Mac sales."

        ``12%`` is *nmod* of ``increase`` (NOUN). From ``increase`` find
        the *prep* child (``in``) and its *pobj* (``Mac sales``) -- that
        prep object is the recorded target. Falls back to the modified
        head noun itself if no prep/pobj exists.
        """
        # Find the head NOUN/PROPN that the percent modifies.
        head_noun: Optional[Token] = None
        anc: Token = ent.root.head
        for _ in range(8):
            if anc.pos_ in {"NOUN", "PROPN"}:
                head_noun = anc
                break
            if anc.head.i == anc.i:
                break
            anc = anc.head

        if head_noun is None:
            return None

        target_text: Optional[str] = None
        # Look for prep -> pobj.
        for child in head_noun.children:
            if child.dep_ == "prep":
                for grand in child.children:
                    if grand.dep_ == "pobj":
                        chunk = self._enclosing_noun_chunk(grand)
                        target_text = (chunk.text if chunk is not None else grand.text).strip()
                        break
            if target_text:
                break

        # Fallback: the modified head noun's own chunk.
        if target_text is None:
            chunk = self._enclosing_noun_chunk(head_noun)
            target_text = (chunk.text if chunk is not None else head_noun.text).strip()

        if not target_text:
            return None

        return {
            "target_text": target_text,
            "entity_label": "PERCENT",
            "numeric_value": ent.text,
            "governing_verb": "",
            "is_financial": True,
            "sentence": sent.text.strip(),
            "component_type": component_type,
        }

    # ------------------------------------------------------------------
    # Strict / legacy extractor (kept for ablation, mode="strict")
    # ------------------------------------------------------------------

    def _extract_strict_mode(
        self, sent: Span, component_type: int
    ) -> List[Dict]:
        """Legacy noun-chunk + numeric + goal-verb extractor."""
        targets: List[Dict] = []

        ent_map: Dict[int, str] = {}
        for ent in sent.ents:
            if ent.label_ in self.ner_labels:
                for tok in ent:
                    ent_map[tok.i] = ent.label_

        for np in sent.noun_chunks:
            target = self._evaluate_noun_phrase(np, ent_map, sent, component_type)
            if target is not None:
                targets.append(target)

        return targets

    def _evaluate_noun_phrase(
        self,
        np: Span,
        ent_map: Dict[int, str],
        sent: Span,
        component_type: int,
    ) -> Optional[Dict]:
        """
        Evaluate one noun phrase against the three legacy criteria.

        Returns a target dict if all criteria are satisfied, else None.
        Used only in ``mode="strict"``.
        """
        # ---- Criterion 1: Contains a relevant Named Entity ----------------
        entity_label: Optional[str] = None
        for tok in np:
            if tok.i in ent_map:
                entity_label = ent_map[tok.i]
                break

        if entity_label is None:
            return None

        # ---- Criterion 2: Modified by a numeric expression ----------------
        numeric_value: Optional[str] = None
        for tok in np:
            for child in tok.children:
                if child.pos_ in NUMERIC_POS or child.dep_ in NUMERIC_DEPS:
                    numeric_value = child.text
                    break
            if numeric_value:
                break

        root = np.root
        if numeric_value is None:
            if root.pos_ in NUMERIC_POS or root.dep_ in NUMERIC_DEPS:
                numeric_value = root.text

        if numeric_value is None:
            for tok in np:
                if tok.pos_ in NUMERIC_POS:
                    numeric_value = tok.text
                    break

        if numeric_value is None:
            return None

        # ---- Criterion 3: Governed by a goal-oriented verb ----------------
        governing_verb: Optional[str] = self._find_governing_verb(root)
        if governing_verb is None:
            return None

        return {
            "target_text": np.text,
            "entity_label": entity_label,
            "numeric_value": numeric_value,
            "governing_verb": governing_verb,
            "is_financial": entity_label in FINANCIAL_LABELS,
            "sentence": sent.text.strip(),
            "component_type": component_type,
        }

    def _find_governing_verb(self, token: Token) -> Optional[str]:
        """
        Walk the dependency tree upward from *token* looking for a goal verb.

        Traversal path: token → head → head … up to sentence root.
        We also check the token's own head's children for subject/object
        relationships (the typical pattern: verb → dobj/nsubj → NP).
        """
        visited: set[int] = set()
        current: Token = token

        while current.i not in visited:
            visited.add(current.i)

            # Check the head of the current token
            head: Token = current.head

            if head.pos_ == "VERB" or head.pos_ == "AUX":
                lemma = head.lemma_.lower()
                if lemma in self.goal_verbs:
                    return lemma

            # Reached the root without finding a match
            if head.i == current.i:
                break

            current = head

        return None


# ---------------------------------------------------------------------------
# __main__ — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )

    print("=== PAPER MODE (Cohen & Nguyen 2024) ===\n")
    extractor = SpacyTargetExtractor(mode="paper")

    # Apple Q4 2019 examples lifted directly from the paper (Figure 1).
    APPLE_Q4_2019 = [
        {
            "component_type": 2,
            "text": (
                "Net income was $1.67 billion. "
                "We saw a 12% year-over-year increase in Mac sales to US education institutions. "
                "Mac demand remained strong throughout the quarter."
            ),
        },
    ]

    targets = extractor.extract_from_transcript(APPLE_Q4_2019)
    print(f"Extracted {len(targets)} performance targets:\n")
    for t in targets:
        print(
            f"  [{t['entity_label']:8s}] target_text={t['target_text']!r:30s} "
            f"numeric={t['numeric_value']!r:18s} "
            f"verb={t['governing_verb']!r}"
        )

    print("\n=== STRICT MODE (legacy goal-verb gate) ===\n")
    strict = SpacyTargetExtractor(mode="strict")
    SAMPLE_TRANSCRIPT = [
        {
            "component_type": 2,
            "text": (
                "We expect to achieve revenue growth of 15 percent this fiscal year. "
                "Our goal is to deliver $2 billion in free cash flow. "
                "Management is committed to maintaining a 40 percent gross margin."
            ),
        },
    ]
    targets = strict.extract_from_transcript(SAMPLE_TRANSCRIPT)
    print(f"Extracted {len(targets)} performance targets:\n")
    for t in targets:
        print(
            f"  [{t['entity_label']:8s}] {t['target_text']!r:50s} "
            f"verb={t['governing_verb']!r:12s} "
            f"num={t['numeric_value']!r}"
        )

    print("\nNormalization examples:")
    for phrase in ["Net income", "Mac sales", "the revenue growth"]:
        print(f"  {phrase!r:40s} → {extractor.normalize_target(phrase)!r}")
