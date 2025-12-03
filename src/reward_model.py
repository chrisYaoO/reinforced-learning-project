import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter
import numpy as np

SENTIMENT_MODEL_PATH = "../models/sentiment_classifier_yelp"  # Path to your fine-tuned sentiment classifier
Tokenizer_MODEL_PATH = "../models/sentiment_classifier_yelp"

INSTRUCTION_KEYWORDS = [
    "write", "say", "generate", "give", "make",
    "produce", "review", "sentence", "comment"
]

def is_instruction(prompt):
    text = prompt.lower()
    return any(k in text for k in INSTRUCTION_KEYWORDS)


class RewardModel:
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 0.1,
                 gamma: float = 0.05,
                 device: str = None):
        """
        Initializes the reward model, which is a weighted sum of three components.
        R(x,y) = α * R_sent - β * R_rep + γ * R_flu

        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. R_sent: Load the sentiment classifier, fine-tune and provide this model
        print(f"Loading sentiment classifier from: {SENTIMENT_MODEL_PATH}")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(Tokenizer_MODEL_PATH)
        print("Sentiment classifier loaded.")

        # 2. R_rep & R_flu: Set n-gram parameters
        self.rep_ngram = 3  # Use trigrams for repetition penalty
        self.flu_ngram = 2  # Use bigrams (Distinct-2) for fluency/diversity calculation
        self.ideal_length = 40 # The ideal response length we desire

        # 3. Coefficients
        # tuning these coefficients
        self.alpha = alpha  # Sentiment
        self.beta = beta    # Repetition penalty
        self.gamma = gamma  # Fluency

    def _compute_r_sent(self, response_text: str) -> float:
        """
        Calculates R_sent (Sentiment Reward).
        Returns a continuous score in the range [-1, 1]
        """
        try:
            inputs = self.tokenizer(response_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits
            
            # Convert logits to probabilities in the [0, 1] range
            probs = F.softmax(logits, dim=-1)
            
            # label 1 is "POSITIVE"
            # For sst-2, [0] is NEGATIVE, [1] is POSITIVE
            prob_positive = probs[0][1].item()
            
            # Map the [0, 1] probability to a [-1, 1] reward score
            # 0 -> -1, 0.5 -> 0, 1 -> 1
            score = (prob_positive * 2) - 1
            return score
        
        except Exception as e:
            print(f"Warning: R_sent calculation failed for text: '{response_text}'. Error: {e}")
            return 0.0

    def _compute_r_rep(self, response_text: str) -> float:
        """
        Calculates R_rep (Repetition Penalty)
        Based on n-gram duplication statistics
        Returns a score in the [0, 1] range, where a higher score means more severe repetition.
        """
        tokens = response_text.lower().split()
        n = self.rep_ngram
        
        if len(tokens) < n:
            return 0.0  # Not enough tokens
        
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        
        if not ngrams:
            return 0.0
        
        # Calculate the "waste" caused by repetition
        # (Total n-grams - Unique n-grams) / Total n-grams
        total_ngrams = sum(ngrams.values())
        unique_ngrams = len(ngrams)
        
        repetition_score = (total_ngrams - unique_ngrams) / total_ngrams
        return repetition_score


    def _compute_r_flu(self, response_text: str) -> float:
        """
        Calculates R_flu (Fluency/Diversity Reward).
        Combines lexical diversity (Distinct-n) and text length
        Returns a score in the [0, 1] range, where higher is better.
        """
        tokens = response_text.lower().split()
        
        # 1. Lexical Diversity (Distinct-2)
        n = self.flu_ngram
        if len(tokens) < n:
            diversity_score = 0.0
        else:
            ngrams = set()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams.add(ngram)
            
            diversity_score = len(ngrams) / (len(tokens) - n + 1) # Normalized
        
        # 2. Text Length Reward (R_len)
        # Reward lengths close to "ideal_length", penalize too short or too long
        length_diff = abs(len(tokens) - self.ideal_length)
        # Use an exponential decay to penalize deviation
        # 1.0 (perfect) -> 0.0 (very bad)
        length_score = np.exp(-0.05 * length_diff) 
        
        # Combine: 50% diversity and 50% length
        fluency_score = (0.5 * diversity_score) + (0.5 * length_score)
        return fluency_score


    def compute_reward(self, prompt: str, response: str) -> (float, float, float, float):
        """
        Calculates the total reward score R(x,y).
        
        Parameters:
        prompt (str): The input prompt (x in your formula)
        response (str): The model-generated response (y in your formula)
        
        Returns:
        tuple: (final_reward, r_sent, r_rep, r_flu)
               Returns all components for logging by the PPO trainer
        """
        
        #
        r_sent = self._compute_r_sent(response)
        
        if is_instruction(prompt):
    
            # ----- Instruction prompts -----
            lower = prompt.lower()
            if "positive" in lower:
                desired_sentiment = 1
            elif "negative" in lower:
                desired_sentiment = 0
            else:
                # instruction but no sentiment target → disable sentiment reward
                desired_sentiment = None

        else:
            # ----- Continuation prompts -----
            prompt_sentiment = self._compute_r_sent(prompt)
            if prompt_sentiment > 0:
                desired_sentiment = 1
            elif prompt_sentiment < 0:
                desired_sentiment = 0
            else:
                desired_sentiment = None
        if desired_sentiment == 0:
            # want negative → flip r_sent
            r_sent = -r_sent

        r_rep = self._compute_r_rep(response)
        r_flu = self._compute_r_flu(response)
        
        final_reward = (self.alpha * r_sent) - (self.beta * r_rep) + (self.gamma * r_flu)
        
        return final_reward, r_sent, r_rep, r_flu

# --- Main execution / test block ---
if __name__ == "__main__":
    print("Initializing Reward Model...")
    #test coefficient tuning here
    reward_model = RewardModel(alpha=1.0, beta=0.1, gamma=0.05)
    
    print("\n--- Test Case 1: Good, fluent response ---")
    prompt1 = "Write a positive one-sentence review:"
    response1 = "The food was absolutely fantastic and the service was just as good!"
    
    rew1, sent1, rep1, flu1 = reward_model.compute_reward(prompt1, response1)
    print(f"Response: '{response1}'")
    print(f"  R_sent (Sentiment): {sent1:.4f} (High is good)")
    print(f"  R_rep (Repetition): {rep1:.4f} (Low is good)")
    print(f"  R_flu (Fluency):    {flu1:.4f} (High is good)")
    print(f"-> FINAL REWARD: {rew1:.4f}")

    print("\n--- Test Case 2: Repetitive, bad response ---")
    prompt2 = "Write a positive one-sentence review:"
    response2 = "bad bad bad bad bad bad bad bad bad bad bad bad"
    
    rew2, sent2, rep2, flu2 = reward_model.compute_reward(prompt2, response2)
    print(f"Response: '{response2}'")
    print(f"  R_sent (Sentiment): {sent2:.4f}")
    print(f"  R_rep (Repetition): {rep2:.4f} (High score = high penalty)")
    print(f"  R_flu (Fluency):    {flu2:.4f}")
    print(f"-> FINAL REWARD: {rew2:.4f}")
    
    print("\n--- Test Case 3: Neutral, short response ---")
    prompt3 = "Write a positive one-sentence review:"
    response3 = "It was okay."
    
    rew3, sent3, rep3, flu3 = reward_model.compute_reward(prompt3, response3)
    print(f"Response: '{response3}'")
    print(f"  R_sent (Sentiment): {sent3:.4f}")
    print(f"  R_rep (Repetition): {rep3:.4f}")
    print(f"  R_flu (Fluency):    {flu3:.4f} (Penalized for being too short)")
    print(f"-> FINAL REWORD: {rew3:.4f}")