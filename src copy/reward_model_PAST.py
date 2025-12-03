import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter
import numpy as np

# 情感分类模型（你自己 finetune 好的路径）
SENTIMENT_MODEL_PATH = "../models/sentiment_classifier_yelp"


class RewardModel:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        device: str | None = None,
    ):
        """
        总体奖励：
            R(x, y) = α * R_sent_aligned - β * R_rep + γ * R_flu

        其中：
            - R_sent_aligned 会根据 prompt 的意图（正/负/中性）动态调整方向
            - R_rep 是重复惩罚（越重复越大）
            - R_flu 是流畅度/多样性奖励
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 1. 情感模型 + 对应 tokenizer
        print(f"Loading sentiment classifier from: {SENTIMENT_MODEL_PATH}")
        try:
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                SENTIMENT_MODEL_PATH
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
            print("Sentiment classifier loaded.")
        except OSError:
            print(f"Error: Could not load model from {SENTIMENT_MODEL_PATH}. Please check the path.")
            raise

        # 记录 num_labels 和“正向标签”的 index
        config = self.sentiment_model.config
        self.num_labels = getattr(config, "num_labels", None) or 1

        # 默认：二分类时假设 id=1 是 POSITIVE
        self.pos_label_idx = 1 if self.num_labels >= 2 else 0

        # 尝试从 config 中自动推断 POSITIVE 的 index
        if hasattr(config, "id2label"):
            try:
                id2label = {int(k): v for k, v in config.id2label.items()}
                for idx, name in id2label.items():
                    name_u = str(name).upper()
                    if "POS" in name_u:
                        self.pos_label_idx = idx
                        break
            except Exception:
                pass

        print(
            f"[RewardModel] num_labels = {self.num_labels}, "
            f"pos_label_idx = {self.pos_label_idx}"
        )

        # 2. n-gram 设置
        self.rep_ngram = 3  # 重复惩罚使用 3-gram
        self.flu_ngram = 2  # 流畅度里 Distinct-2
        self.ideal_length = 40  # 期望长度

        # 3. 系数（可做 ablation）
        self.alpha = alpha  # Sentiment coeff
        self.beta = beta    # Repetition penalty coeff
        self.gamma = gamma  # Fluency coeff

    # ===== 判定 prompt 情感目标：positive / negative / neutral =====
    def _detect_prompt_polarity(self, prompt: str) -> str:
        """
        根据 prompt 文本粗略判断用户需要的情感方向：
            - "positive": 正向/暖心/推荐/兴奋
            - "negative": 投诉/警告/卫生/价格不满等
            - "neutral": 说明/客观描述/打分/便利性等
        """
        p = prompt.lower()

        negative_keywords = [
            "negative",
            "complaint",
            "terrible",
            "awful",
            "bad",
            "poor",
            "hate",
            "hygiene issues",
            "cautionary",
            "warning",
            "dirty",
            "overpriced",
            "too expensive",
            "price complaint",
        ]

        positive_keywords = [
            "positive",
            "enthusiastic",
            "excited",
            "exciting",
            "heartwarming",
            "recommend",
            "praise",
            "love",
            "delicious",
        ]

        neutral_keywords = [
            "neutral tone",
            "neutral",
            "objectively",
            "objective",
            "describe",
            "rate",
            "briefly state your opinion",
            "takeout service",
            "parking",
            "accessibility",
            "focusing on convenience",
        ]

        # 优先判断负面
        if any(k in p for k in negative_keywords):
            return "negative"
        else: return "positive"
        # # 再判断正面
        # if any(k in p for k in positive_keywords):
        #     return "positive"
        # # 再判断中性
        # if any(k in p for k in neutral_keywords):
        #     return "neutral"
        # # 默认偏正向（大部分 review task 在 sentiment RLHF 中都是倾向正向）
        # return "positive"

    # ===== R_sent: 先算 Positive 概率，再对齐到 [-1, 1] =====
    def _compute_pos_prob(self, response_text: str) -> float:
        """
        计算 response 是 'Positive' 的概率 (0.0 ~ 1.0)
        """
        try:
            inputs = self.tokenizer(
                response_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits  # [B, num_labels]

            # 保证维度
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            # 获取 Positive 的概率
            if self.num_labels == 1:
                # 回归/单 logit：经 sigmoid
                prob_positive = torch.sigmoid(logits)[0, 0].item()
            else:
                # 多分类：softmax 后取 POS 标签
                probs = F.softmax(logits, dim=-1)
                C = probs.size(-1)
                idx = min(max(self.pos_label_idx, 0), C - 1)
                prob_positive = probs[0, idx].item()

            return float(prob_positive)

        except Exception as e:
            print(f"Warning: R_sent calculation failed for text: '{response_text}'. Error: {e}")
            # 失败时返回中性概率，避免奖励爆炸
            return 0.5

    # ===== R_rep =====
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

        if not ngrams:
            return 0.0

        total_ngrams = sum(ngrams.values())
        unique_ngrams = len(ngrams)

        repetition_score = (total_ngrams - unique_ngrams) / total_ngrams
        return float(repetition_score)

    # ===== R_flu =====
    def _compute_r_flu(self, response_text: str) -> float:
        """
        流畅度 / 多样性奖励，组合 Distinct-2 和 长度奖励，返回 [0,1]
        """
        tokens = response_text.lower().split()

        # 1. Distinct-2
        n = self.flu_ngram
        if len(tokens) < n:
            diversity_score = 0.0
        else:
            ngrams = set()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i: i + n])
                ngrams.add(ngram)
            diversity_score = len(ngrams) / (len(tokens) - n + 1)

        # 2. 长度奖励 (Gaussian decay around ideal_length)
        length_diff = abs(len(tokens) - self.ideal_length)
        length_score = float(np.exp(-0.05 * length_diff))

        fluency_score = 0.5 * diversity_score + 0.5 * length_score
        return float(fluency_score)

    # ===== 总 Reward：考虑 prompt 意图对齐 =====
    def compute_reward(self, prompt: str, response: str) -> tuple[float, float, float, float]:
        """
        返回: (final_reward, r_sent_aligned, r_rep, r_flu)

        - r_sent_aligned: 根据 prompt 的情感目标 (pos/neg/neutral) 对齐后的情感分，范围 [-1, 1]
        - r_rep: 重复惩罚 [0, 1]
        - r_flu: 流畅度/多样性 [0, 1]
        """
        # 1. 判断 prompt 情感目标
        mode = self._detect_prompt_polarity(prompt)  # "positive" / "negative" / "neutral"

        # 2. 计算 response 是 "Positive" 的概率 (0.0 ~ 1.0)
        pos_prob = self._compute_pos_prob(response)

        # 映射到 [-1, 1] 的 "正向情感分"
        # pos_prob = 1.0 → +1.0
        # pos_prob = 0.0 → -1.0
        base_sent = pos_prob * 2.0 - 1.0

        # 3. 根据 prompt 意图做对齐
        if mode == "positive":
            # 希望输出正向：base_sent 越大越好
            r_sent_aligned = base_sent
        elif mode == "negative":
            # 希望输出负向：base_sent 越小越好 → 取反
            r_sent_aligned = -base_sent
        else:
            # 中性：希望情感接近 0，过正/过负都扣分
            # base_sent 越接近 0 越好：
            #   base_sent = 0   → r = 0
            #   base_sent = ±1 → r = -1
            r_sent_aligned = -abs(base_sent)

        # 4. 其他两项
        r_rep = self._compute_r_rep(response)
        r_flu = self._compute_r_flu(response)

        # 5. 最终 reward
        final_reward = (self.alpha * r_sent_aligned) - (self.beta * r_rep) + (self.gamma * r_flu)

        return float(final_reward), float(r_sent_aligned), float(r_rep), float(r_flu)


# 简单自测
if __name__ == "__main__":
    print("Initializing Reward Model...")
    try:
        reward_model = RewardModel(alpha=1.0, beta=0.3, gamma=0.1)
    except Exception as e:
        print("Model path not found or load error, strictly checking code logic only.")
        print("Error:", e)
        raise SystemExit

    print("\n--- Test Case 1: Prompt asks for POSITIVE, Response is POSITIVE ---")
    prompt1 = "Write a positive one-sentence review:"
    response1 = "The food was absolutely fantastic and the service was just as good!"
    rew1, sent1, rep1, flu1 = reward_model.compute_reward(prompt1, response1)
    print(f"Prompt: {prompt1}")
    print(f"Response: {response1}")
    print(f" -> R_sent_aligned (expect high positive): {sent1:.4f}")
    print(f" -> Final Reward: {rew1:.4f}")

    print("\n--- Test Case 2: Prompt asks for NEGATIVE, Response is NEGATIVE ---")
    prompt2 = "Write a strong negative complaint about the terrible service:"
    response2 = "The service was awful, the staff were rude, and I will never come back."
    rew2, sent2, rep2, flu2 = reward_model.compute_reward(prompt2, response2)
    print(f"Prompt: {prompt2}")
    print(f"Response: {response2}")
    print(f" -> R_sent_aligned (expect high positive because matches negative intent): {sent2:.4f}")
    print(f" -> Final Reward: {rew2:.4f}")

    print("\n--- Test Case 3: Prompt asks for NEGATIVE, Response is POSITIVE ---")
    prompt3 = "Write a negative review:"
    response3 = "I absolutely loved everything about this place, it was perfect!"
    rew3, sent3, rep3, flu3 = reward_model.compute_reward(prompt3, response3)
    print(f"Prompt: {prompt3}")
    print(f"Response: {response3}")
    print(f" -> R_sent_aligned (expect negative, mismatch with intent): {sent3:.4f}")
    print(f" -> Final Reward: {rew3:.4f}")

    print("\n--- Test Case 4: Prompt asks for NEUTRAL, Response is very emotional ---")
    prompt4 = "Objectively describe the dishes and environment of a Chinese restaurant, maintaining a neutral tone."
    response4 = "The food was insanely amazing and I was incredibly happy the whole time!"
    rew4, sent4, rep4, flu4 = reward_model.compute_reward(prompt4, response4)
    print(f"Prompt: {prompt4}")
    print(f"Response: {response4}")
    print(f" -> R_sent_aligned (expect negative because too emotional for neutral): {sent4:.4f}")
    print(f" -> Final Reward: {rew4:.4f}")