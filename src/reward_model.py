# reward_model.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter
import numpy as np

SENTIMENT_MODEL_PATH = "../models/sentiment_classifier_yelp"


class RewardModel:
    """
    总体奖励：
        R(x, y) = α * R_sent_aligned
                  - β * R_rep
                  + γ * R_flu
                  + δ * R_task

    - R_sent_aligned: 基于 prompt 的情感目标（正/负/中性）对齐后的情感分 ∈ [-1, 1]
    - R_rep: 重复惩罚，3-gram 级别，越重复越大 ∈ [0, 1]
    - R_flu: 流畅度/多样性奖励，考虑 Distinct-2 + 长度 ∈ [0, 1]
    - R_task: lexical constraints，对任务相关关键词的命中程度 ∈ [0, 1]
    """

    def __init__(
        self,
        alpha: float = 0.7,   # 情感
        beta: float = 0.6,    # 重复惩罚
        gamma: float = 0.3,   # 流畅度/多样性
        delta: float = 0.3,   # 任务关键词
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

        # n-gram 相关设置
        self.rep_ngram = 3
        self.flu_ngram = 2
        self.ideal_length = 40

        # 系数
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    # ========= Prompt 情感/任务解析 =========

    def _detect_prompt_polarity(self, prompt: str) -> str:
        """
        判定 prompt 的情感目标：
            - "positive": 希望输出正向内容
            - "negative": 希望输出负向/投诉/警告
            - "neutral": 期望客观中性
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

        # 默认：中性，不强行偏正
        return "neutral"

    def _detect_task_keywords(self, prompt: str) -> list[str]:
        """
        基于 prompt 的语义，返回该任务关心的一组关键词（lexical constraints）
        用于 R_task。
        """
        p = prompt.lower()

        # 卫生 / 卫生警告
        if "hygiene" in p or "dirty" in p or "cleanliness" in p:
            return ["dirty", "unclean", "unsanitary", "hair", "smell", "smelly", "stain", "mold"]

        # 价格相关投诉
        if "price" in p or "overpriced" in p or "too expensive" in p:
            return ["expensive", "overpriced", "too much", "pricey", "not worth", "rip-off"]

        # 停车 / 便利性
        if "parking" in p or "accessibility" in p:
            return ["parking", "garage", "lot", "street parking", "accessible", "wheelchair", "stairs", "elevator"]

        # 咖啡不好但甜点好
        if "coffee" in p and "dessert" in p:
            return ["bad coffee", "weak coffee", "bitter coffee", "excellent dessert", "great dessert", "cake", "pastry"]

        # takeout / 外卖便利性
        if "takeout" in p or "take-out" in p:
            return ["takeout", "take-out", "to-go", "pickup", "delivery", "convenient"]

        # 默认不加任务词约束
        return []

    # ========= R_sent 相关 =========

    def _compute_pos_prob(self, response_text: str) -> float:
        """
        用 finetune 的 classifier 计算 P(positive)，范围 [0,1]
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
        根据 prompt 的情感目标对齐情感得分，范围 [-1,1]
        """
        mode = self._detect_prompt_polarity(prompt)
        pos_prob = self._compute_pos_prob(response)
        base_sent = pos_prob * 2.0 - 1.0  # [0,1] -> [-1,1]

        if mode == "positive":
            # 越 positive 越好
            return float(base_sent)
        elif mode == "negative":
            # 越 negative 越好 → 取反
            return float(-base_sent)
        else:
            # neutral: 越接近中性越好
            # pos_prob=0.5 时最好，使用抛物线：
            # r = -4 * (p-0.5)^2 + 1  ∈ [-0,1]
            r = -4.0 * (pos_prob - 0.5) ** 2 + 1.0
            return float(max(-1.0, min(1.0, r)))

    # ========= R_rep =========

    def _compute_r_rep(self, response_text: str) -> float:
        """
        n-gram 重复惩罚，返回 [0,1]，越大重复越严重
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
        流畅度/多样性：结合 Distinct-2 + 长度
        返回 [0,1]
        """
        tokens = response_text.lower().split()

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

        # 长度奖励（在 ideal_length ≈40 左右最好）
        length_diff = abs(len(tokens) - self.ideal_length)
        length_score = float(np.exp(-0.05 * length_diff))

        fluency_score = 0.5 * diversity_score + 0.5 * length_score
        return float(fluency_score)

    # ========= R_task =========

    def _compute_r_task(self, prompt: str, response: str) -> float:
        """
        lexical constraints: 根据 prompt 需要的关键词，对 response 打额外奖励。
        0 ~ 1，命中关键词越多，得分越高。
        """
        response_lower = response.lower()
        keywords = self._detect_task_keywords(prompt)
        if not keywords:
            return 0.0

        hits = sum(1 for kw in keywords if kw in response_lower)
        if hits == 0:
            return 0.0

        # 简单归一化：命中 1 个 → 0.5，命中 >=2 → 1.0
        score = min(1.0, hits * 0.5)
        return float(score)

    # ========= 总 reward =========

    def compute_reward(
        self, prompt: str, response: str
    ) -> tuple[float, float, float, float, float]:
        """
        返回:
          - final_reward
          - r_sent_aligned
          - r_rep
          - r_flu
          - r_task
        """
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
         
    # ]
    tests = [
    # ===== 正向任务：response 强正向 → r_sent_aligned 应为正 =====
    (
        "Write a positive one-sentence review:",
        "The food was absolutely fantastic and the service was great!"
    ),

    # ===== 负向任务：response 强负向 → r_sent_aligned 应为正（匹配负向） =====
    (
        "Write a strong negative complaint about the terrible service:",
        "The service was awful and the staff were extremely rude."
    ),

    # ===== 中性任务：response 中性客观 → r_sent_aligned 应为高分（接近 1） =====
    (
        "Objectively describe the dishes and environment of a Chinese restaurant, maintaining a neutral tone.",
        "The restaurant has bright lighting, wooden tables, and the dishes are served in simple white plates."
    ),

    # ===== 负向任务：卫生问题 → response 负向 → r_sent_aligned 应为正（匹配负向） =====
    (
        "Write a cautionary review about a restaurant's hygiene issues.",
        "The restroom was dirty and the tables were sticky; it really felt unsanitary."
    ),

    # # ===== 负向任务：价格抱怨 → response 负向 → r_sent_aligned 应为正（匹配负向） =====
    # (
    #     "You dined at an upscale restaurant. Write a complaint about the price.",
    #     "The food was good, but the dishes were overpriced and not worth the money."
    # ),
        # ===== 负向任务：价格抱怨 → response 负向 → r_sent_aligned 应为正（匹配负向） =====
    (
        "You dined at an upscale restaurant. Write a complaint about the price.",
        "The food was absolutely fantastic and the service was great! "
    ),
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
