# reward_model.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter
import numpy as np

SENTIMENT_MODEL_PATH = "../models/sentiment_classifier_yelp"


class RewardModel:
    """
    Overall reward:
        R(x, y) = α * R_sent_aligned
                  - β * R_rep
                  + γ * R_flu
                  + δ * R_task

    - R_sent_aligned: sentiment score aligned with the sentiment target of the prompt
      (positive/negative/neutral) ∈ [-1, 1]
    - R_rep: repetition penalty at the 3-gram level, higher means more repetition ∈ [0, 1]
    - R_flu: fluency/diversity reward, considering Distinct-2 + length ∈ [0, 1]
    - R_task: lexical constraints, measures how well task-related keywords are hit ∈ [0, 1]
    """

    def __init__(
        self,
        alpha: float = 0.7,   # sentiment
        beta: float = 0.4,    # repetition penalty
        gamma: float = 0.3,   # fluency/diversity
        delta: float = 0.3,   # task keywords
        device: str | None = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading sentiment classifier from: {SENTIMENT_MODEL_PATH}")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL_PATH
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
        print("Sentiment classifier loaded.")

        config = self.sentiment_model.config
        self.num_labels = getattr(config, "num_labels", None) or 2
        self.pos_label_idx = 1 if self.num_labels >= 2 else 0

        if hasattr(config, "id2label"):
            try:
                id2label = {int(k): v for k, v in config.id2label.items()}
                for idx, name in id2label.items():
                    if "POS" in str(name).upper():
                        self.pos_label_idx = idx
                        break
            except Exception:
                pass

        print(
            f"[RewardModel] num_labels = {self.num_labels}, "
            f"pos_label_idx = {self.pos_label_idx}"
        )

        # n-gram related settings
        self.rep_ngram = 3
        self.flu_ngram = 2
        self.ideal_length = 40

        # coefficients
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    # ========= Prompt sentiment / task parsing =========

    def _detect_prompt_polarity(self, prompt: str) -> str:
        """
        Determine the sentiment target of the prompt:
            - "positive": wants a positive output
            - "negative": wants a negative/complaint/warning output
            - "neutral": expects objective/neutral content
        """
        p = prompt.lower()

        negative_keywords = [
            "negative", "complaint", "terrible", "awful", "bad", "poor",
            "hate", "hygiene issues", "cautionary", "warning", "dirty",
            "overpriced", "too expensive", "price complaint",
        ]

        positive_keywords = [
            "positive", "enthusiastic", "excited", "exciting",
            "heartwarming", "recommend", "praise", "love", "delicious",
            "uplifting", "cheerful",
        ]

        neutral_keywords = [
            "neutral tone", "neutral", "objectively", "objective",
            "describe", "rate", "briefly state your opinion",
            "parking", "accessibility", "takeout service",
            "focusing on convenience",
        ]

        if any(k in p for k in negative_keywords):
            return "negative"
        if any(k in p for k in positive_keywords):
            return "positive"
        if any(k in p for k in neutral_keywords):
            return "neutral"

        # Default: neutral, do not force it to be positive or negative
        return "neutral"

    def _detect_task_keywords(self, prompt: str) -> list[str]:
        """
        Based on the semantics of the prompt, return a set of task-specific keywords
        (lexical constraints) used for R_task.
        """
        p = prompt.lower()

        # hygiene / cleanliness warnings
        if "hygiene" in p or "dirty" in p or "cleanliness" in p:
            return ["dirty", "unclean", "unsanitary", "hair", "smell", "smelly", "stain", "mold"]

        # price-related complaints
        if "price" in p or "overpriced" in p or "too expensive" in p:
            return ["expensive", "overpriced", "too much", "pricey", "not worth", "rip-off"]

        # parking / accessibility
        if "parking" in p or "accessibility" in p:
            return ["parking", "garage", "lot", "street parking", "accessible", "wheelchair", "stairs", "elevator"]

        # bad coffee but good dessert
        if "coffee" in p and "dessert" in p:
            return ["bad coffee", "weak coffee", "bitter coffee", "excellent dessert", "great dessert", "cake", "pastry"]

        # takeout / to-go convenience
        if "takeout" in p or "take-out" in p:
            return ["takeout", "take-out", "to-go", "pickup", "delivery", "convenient"]

        # By default, no task keyword constraints
        return []

    # ========= R_sent related =========

    def _compute_pos_prob(self, response_text: str) -> float:
        """
        Use the fine-tuned classifier to compute P(positive), range [0, 1]
        """
        try:
            inputs = self.tokenizer(
                response_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.sentiment_model(**inputs).logits

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            if self.num_labels == 1:
                prob_positive = torch.sigmoid(logits)[0, 0].item()
            else:
                probs = F.softmax(logits, dim=-1)
                C = probs.size(-1)
                idx = min(max(self.pos_label_idx, 0), C - 1)
                prob_positive = probs[0, idx].item()

            return float(prob_positive)
        except Exception as e:
            print(f"[RewardModel] Warning in _compute_pos_prob: {e}")
            return 0.5

    def _compute_r_sent_aligned(self, prompt: str, response: str) -> float:
        """
        Compute the sentiment score aligned with the prompt's sentiment target, in [-1, 1]
        """
        mode = self._detect_prompt_polarity(prompt)
        pos_prob = self._compute_pos_prob(response)
        base_sent = pos_prob * 2.0 - 1.0  # [0,1] -> [-1,1]

        if mode == "positive":
            # More positive is better
            return float(base_sent)
        elif mode == "negative":
            # More negative is better → invert
            return float(-base_sent)
        else:
            # neutral: being closer to neutral is better
            # pos_prob=0.5 is optimal, use a parabola:
            # r = -4 * (p-0.5)^2 + 1  ∈ [-0,1]
            r = -4.0 * (pos_prob - 0.5) ** 2 + 1.0
            return float(max(-1.0, min(1.0, r)))

    # ========= R_rep =========

    def _compute_r_rep(self, response_text: str) -> float:
        """
        n-gram repetition penalty, returns [0,1]; higher means more severe repetition
        """
        tokens = response_text.lower().split()
        n = self.rep_ngram

        if len(tokens) < n:
            return 0.0

        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i: i + n])
            ngrams[ngram] += 1

        total_ngrams = sum(ngrams.values())
        unique_ngrams = len(ngrams)
        if total_ngrams == 0:
            return 0.0

        repetition_score = (total_ngrams - unique_ngrams) / total_ngrams
        return float(repetition_score)

    # ========= R_flu =========
    def _compute_r_flu(self, response_text: str) -> float:
        """
        Fluency/diversity: combine Distinct-2 + length.
        Returns [0,1].
        """
        tokens = response_text.lower().split()

        # No tokens at all: return 0 directly, no fluency reward
        if len(tokens) == 0:
            return 0.0

        # Distinct-2
        n = self.flu_ngram
        if len(tokens) < n:
            diversity_score = 0.0
        else:
            ngrams = set()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i: i + n])
                ngrams.add(ngram)
            diversity_score = len(ngrams) / (len(tokens) - n + 1)

        # Length reward (best around ideal_length ≈ 40)
        length_diff = abs(len(tokens) - self.ideal_length)
        length_score = float(np.exp(-0.05 * length_diff))

        fluency_score = 0.5 * diversity_score + 0.5 * length_score
        return float(fluency_score)

    # ========= R_task =========

    def _compute_r_task(self, prompt: str, response: str) -> float:
        """
        Lexical constraints: give extra reward based on keywords required by the prompt.
        Range 0 ~ 1; the more keywords hit, the higher the score.
        """
        response_lower = response.lower()
        keywords = self._detect_task_keywords(prompt)
        if not keywords:
            return 0.0

        hits = sum(1 for kw in keywords if kw in response_lower)
        if hits == 0:
            return 0.0

        # Simple normalization: hit 1 keyword → 0.5, hit >= 2 keywords → 1.0
        score = min(1.0, hits * 0.5)
        return float(score)

    # ========= Overall reward =========
    # With an additional check for empty output
    def compute_reward(
        self, prompt: str, response: str
    ) -> tuple[float, float, float, float, float]:
        """
        Returns:
          - final_reward
          - r_sent_aligned
          - r_rep
          - r_flu
          - r_task
        """

        # === Special case: response is empty or whitespace only, apply strong penalty ===
        clean_resp = (response or "").strip()
        if len(clean_resp) == 0:
            # Empty output: treat as worst case
            r_sent = -1.0          # strongly misaligned with any task
            r_rep = 0.0            # no content → no repetition
            r_flu = 0.0            # no content → no fluency
            r_task = 0.0           # no content → cannot hit task keywords

            final_reward = (
                self.alpha * r_sent
                - self.beta * r_rep
                + self.gamma * r_flu
                + self.delta * r_task
            )
            # Here final_reward = -alpha, clearly a very low negative score
            return float(final_reward), float(r_sent), float(r_rep), float(r_flu), float(r_task)

        # === Normal case: non-empty response goes through standard scoring logic ===
        r_sent = self._compute_r_sent_aligned(prompt, response)
        r_rep = self._compute_r_rep(response)
        r_flu = self._compute_r_flu(response)
        r_task = self._compute_r_task(prompt, response)

        final_reward = (
            self.alpha * r_sent
            - self.beta * r_rep
            + self.gamma * r_flu
            + self.delta * r_task
        )

        return float(final_reward), float(r_sent), float(r_rep), float(r_flu), float(r_task)


if __name__ == "__main__":
    print("Initializing RewardModel for quick sanity check...")
    rm = RewardModel()

    # tests = [
    #     ("Write a positive one-sentence review:", "The food was absolutely fantastic and the service was great!"),
    #     ("Write a strong negative complaint about the terrible service:", "The service was awful and the staff were extremely rude."),
    #     ("Objectively describe the dishes and environment of a Chinese restaurant, maintaining a neutral tone.",
    #      "The restaurant has bright lighting, wooden tables, and the dishes are served in simple white plates."),
    #     ("Write a cautionary review about a restaurant's hygiene issues.",
    #      "The restroom was dirty and the tables were sticky; it really felt unsanitary."),
    #     ("You dined at an upscale restaurant. Write a complaint about the price.",
    #      "The food was good, but the dishes were overpriced and not worth the money."),
    #
    # ]
    tests = [
        # ===== output = "" =====
        (
            "Write a positive one-sentence review:",
            ""
        ),
        # ===== Positive task: strongly positive response → r_sent_aligned should be positive =====
        (
            "Write a positive one-sentence review:",
            "The food was absolutely fantastic and the service was great!"
        ),

        # ===== Negative task: strongly negative response → r_sent_aligned should be positive (matches negative) =====
        (
            "Write a strong negative complaint about the terrible service:",
            "The service was awful and the staff were extremely rude."
        ),

        # ===== Neutral task: neutral, objective response → r_sent_aligned should be high (close to 1) =====
        (
            "Objectively describe the dishes and environment of a Chinese restaurant, maintaining a neutral tone.",
            "The restaurant has bright lighting, wooden tables, and the dishes are served in simple white plates."
        ),

        # ===== Negative task: hygiene issues → negative response → r_sent_aligned should be positive (matches negative) =====
        (
            "Write a cautionary review about a restaurant's hygiene issues.",
            "The restroom was dirty and the tables were sticky; it really felt unsanitary."
        ),

        # # ===== Negative task: price complaint → negative response → r_sent_aligned should be positive (matches negative) =====
        # (
        #     "You dined at an upscale restaurant. Write a complaint about the price.",
        #     "The food was good, but the dishes were overpriced and not worth the money."
        # ),
        # ===== Negative task: price complaint → positive response → r_sent_aligned should be negative (mismatched polarity) =====
        (
            "You dined at an upscale restaurant. Write a complaint about the price.",
            "The food was absolutely fantastic and the service was great! "
        )
    ]

    for p, r in tests:
        fr, rs, rr, rf, rt = rm.compute_reward(p, r)
        print("=" * 80)
        print("PROMPT :", p)
        print("RESP   :", r)
        print(f"  R_sent_aligned = {rs:.3f}")
        print(f"  R_rep          = {rr:.3f}")
        print(f"  R_flu          = {rf:.3f}")
        print(f"  R_task         = {rt:.3f}")
        print(f"  FINAL_REWARD   = {fr:.3f}")
