from transformers import AutoModelForCasualLM, AutoTokenizer
from transformers import pipeline

model=AutoModelForCasualLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

messages= [{"role":"user",
            "content":"Create a funny joke about chickens."}]

output=generator(messages)
print(output[0]["generated_text"])

