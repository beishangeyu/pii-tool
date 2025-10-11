from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1️⃣ 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bigcode/starpii")
model = AutoModelForTokenClassification.from_pretrained("bigcode/starpii")

# 2️⃣ 构建推理管线（pipeline）
pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    device=3,  # 如果没有GPU请改成 -1
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
        "entity": "B-NAME",
        "score": 0.99984944,
        "index": 3,
        "word": "ĠJohn",
        "start": 10,
        "end": 15,
    },
    {
        "entity": "I-NAME",
        "score": 0.9998228,
        "index": 4,
        "word": "ĠDoe",
        "start": 15,
        "end": 19,
    },
    {
        "entity": "B-EMAIL",
        "score": 0.9997565,
        "index": 9,
        "word": "Ġj",
        "start": 35,
        "end": 37,
    },
    {
        "entity": "I-EMAIL",
        "score": 0.99989986,
        "index": 10,
        "word": "ohn",
        "start": 37,
        "end": 40,
    },
    {
        "entity": "I-EMAIL",
        "score": 0.9999218,
        "index": 11,
        "word": ".",
        "start": 40,
        "end": 41,
    },
    {
        "entity": "I-EMAIL",
        "score": 0.9999194,
        "index": 12,
        "word": "doe",
        "start": 41,
        "end": 44,
    },
    {
        "entity": "I-EMAIL",
        "score": 0.99992037,
        "index": 13,
        "word": "@",
        "start": 44,
        "end": 45,
    },
    {
        "entity": "I-EMAIL",
        "score": 0.99992216,
        "index": 14,
        "word": "example",
        "start": 45,
        "end": 52,
    },
    {
        "entity": "I-EMAIL",
        "score": 0.9999236,
        "index": 15,
        "word": ".",
        "start": 52,
        "end": 53,
    },
    {
        "entity": "I-EMAIL",
        "score": 0.99992657,
        "index": 16,
        "word": "com",
        "start": 53,
        "end": 56,
    },
]
[]
