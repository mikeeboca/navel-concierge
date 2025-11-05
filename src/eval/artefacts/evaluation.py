import re
from typing import List, Dict, Any

# Bewertung generierter Antworten gegen Fakten
# Deprecated: Klasse wird nicht mehr aktiv verwendet
class Evaluation:
    def __init__(self, answer: str, cards: List[Dict[str, Any]], question: str):
        self.answer = answer.strip()
        self.cards = cards
        self.question = question
        self.facts_text = self._get_facts_as_single_string()

    def _get_facts_as_single_string(self) -> str:
        text = ""
        for card in self.cards:
            text += f"Titel: {card.get('title', '')}. "
            text += f"Ausschnitt: {card.get('snippet', '')}. "
            text += " ".join(card.get('facts', [])) + ". "
        return text.lower()

    def _extract_terms(self, text: str) -> List[str]:
        text = text.lower()
        return [
            t for t in re.split(r'\s+|-|\?|\!', text)
            if len(t) >= 3 and t.isalpha()
        ]

    def check_for_consistency(self) -> Dict[str, Any]:
        # Satzweise Abgleich mit Faktext
        sentences = re.split(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s',
            self.answer
        )
        sentences = [s.strip() for s in sentences if s.strip()]

        if not self.facts_text:
            return {"score": 0.0, "reason": "keine Fakten vorhanden"}

        unverified = []
        verified_count = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            terms = self._extract_terms(sentence_lower)
            if any(term in self.facts_text for term in terms):
                verified_count += 1
            else:
                unverified.append(sentence)

        score = verified_count / len(sentences) if sentences else 0.0
        is_consistent = score > 0.5

        return {
            "is_consistent": is_consistent,
            "score": round(score, 2),
            "unverified_sentences": unverified
        }

    def check_for_relevance(self) -> bool:
        # Frage–Antwort–Overlap
        question_terms = set(self._extract_terms(self.question))
        answer_terms = set(self._extract_terms(self.answer))

        if not question_terms:
            return True

        return any(term in answer_terms for term in question_terms)

    def evaluate(self) -> Dict[str, Any]:
        return {
            "consistency": self.check_for_consistency(),
            "relevance": self.check_for_relevance()
        }
