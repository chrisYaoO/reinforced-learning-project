import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 1. 配置路径 (必须与训练代码中的 OUTPUT_DIR 一致)
MODEL_PATH = "../models/sentiment_classifier_yelp"

def main():
    # 2. 加载模型和分词器
    print(f"正在从 {MODEL_PATH} 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"❌ 错误: 无法在 {MODEL_PATH} 找到模型。请确保训练脚本已运行完成，且路径正确。")
        return

    # 将模型设置为评估模式 (对于 Dropout 等层很重要)
    model.eval()

    # 定义标签映射 (Yelp Polarity 数据集: 0=负面, 1=正面)
    id2label = {0: "👎 负面 (Negative)", 1: "👍 正面 (Positive)"}

    # 3. 定义预测函数
    def predict_sentiment(text):
        # 预处理输入文本
        inputs = tokenizer(
            text, 
            return_tensors="pt",  # 返回 PyTorch 张量
            truncation=True, 
            max_length=256, 
            padding=True
        )

        # 禁用梯度计算以节省内存并加速推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取 Logits (原始输出)
        logits = outputs.logits
        
        # 使用 Softmax 将 Logits 转换为概率 (0.0 - 1.0)
        probabilities = F.softmax(logits, dim=-1)
        
        # 获取概率最大的类别 ID
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        
        # 获取该类别的置信度分数
        confidence = probabilities[0][predicted_class_id].item()
        
        return id2label[predicted_class_id], confidence

    # 4. 运行预设的测试用例
    print("\n" + "="*40)
    print("   🤖 自动测试预设案例")
    print("="*40)
    
    test_sentences = [
        "The food was absolutely delicious and the service was great!",  # 明显正面
        "I waited for an hour and the pasta was cold. Terrible.",       # 明显负面
        "It was okay, not the best but not the worst.",                 # 中性/模糊
        "The ambiance is nice, but the food is overpriced.",            # 混合评价
        "I will definitely come back again.",                           # 正面意图
    ]

    for text in test_sentences:
        label, score = predict_sentiment(text)
        print(f"\n📝 文本: {text}")
        print(f"🔮 预测: {label}")
        print(f"📊 置信度: {score:.4f}")

    # 5. 交互模式 (手动输入)
    print("\n" + "="*40)
    print("   ⌨️  交互测试模式 (输入 'q' 退出)")
    print("="*40)
    
    while True:
        user_input = input("\n请输入一句英文评论: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("再见！👋")
            break
        
        if not user_input.strip():
            continue

        label, score = predict_sentiment(user_input)
        print(f" -> 预测结果: {label} (置信度: {score:.2%})")

if __name__ == "__main__":
    main()