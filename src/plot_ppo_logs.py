import json
import matplotlib.pyplot as plt

with open("ppo_training_logs.json", "r") as f:
    logs = json.load(f)

iters = logs["iteration"]
reward = logs["reward"]
kl = logs["kl"]
length = logs["length"]
distinct1 = logs["distinct_1"]

# ----- Figure 1: mean reward -----
plt.figure(figsize=(7,5))
plt.plot(iters, reward)
plt.xlabel("Iteration")
plt.ylabel("Mean Reward")
plt.title("RLHF Training - Mean Reward per Iteration")
plt.grid(True)
plt.savefig("reward_curve.png")
plt.close()

# ----- Figure 2: KL -----
plt.figure(figsize=(7,5))
# --- Align lengths ---
min_len = min(len(iters), len(kl))
iters_aligned = iters[:min_len]
kl_aligned = kl[:min_len]

plt.plot(iters_aligned, kl_aligned)

plt.xlabel("Iteration")
plt.ylabel("KL Divergence")
plt.title("Policy KL Divergence per Iteration")
plt.grid(True)
plt.savefig("kl_curve.png")
plt.close()

# ----- Figure 3: Length -----
plt.figure(figsize=(7,5))
plt.plot(iters, length)
plt.xlabel("Iteration")
plt.ylabel("Average Response Length")
plt.title("Response Length per Iteration")
plt.grid(True)
plt.savefig("length_curve.png")
plt.close()

# ----- Figure 4: Distinct-1 -----
plt.figure(figsize=(7,5))
plt.plot(iters, distinct1)
plt.xlabel("Iteration")
plt.ylabel("Distinct-1")
plt.title("Response Diversity (Distinct-1)")
plt.grid(True)
plt.savefig("distinct1_curve.png")
plt.close()

print("All plots saved.")