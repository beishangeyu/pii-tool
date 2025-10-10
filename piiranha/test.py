from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1️⃣ 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    "iiiorg/piiranha-v1-detect-personal-information"
)
model = AutoModelForTokenClassification.from_pretrained(
    "iiiorg/piiranha-v1-detect-personal-information"
)

# 2️⃣ 构建推理管线（pipeline）
pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    device=2,  # 如果没有GPU请改成 -1
)

# 3️⃣ 输入要检测的文本
text = [
    "My name is John Doe and my email is john.doe@example.com. 电话: 13800138000",
    "Joe lives in New York.",
]

# 4️⃣ 运行推理
results = pipe(text)

# 5️⃣ 打印结果
for entity in results:
    print(entity)


# example output
[
    {
        "entity": "I-USERNAME",
        "score": 0.53386587,
        "index": 11,
        "word": "▁john",
        "start": 35,
        "end": 40,
    },
    {
        "entity": "I-USERNAME",
        "score": 0.73129725,
        "index": 13,
        "word": "do",
        "start": 41,
        "end": 43,
    },
    {
        "entity": "I-USERNAME",
        "score": 0.6357182,
        "index": 14,
        "word": "e",
        "start": 43,
        "end": 44,
    },
]
[
    {
        "entity": "I-CITY",
        "score": 0.99655545,
        "index": 4,
        "word": "▁New",
        "start": 12,
        "end": 16,
    },
    {
        "entity": "I-CITY",
        "score": 0.9961559,
        "index": 5,
        "word": "▁York",
        "start": 16,
        "end": 21,
    },
]
