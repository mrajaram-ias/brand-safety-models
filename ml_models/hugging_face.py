from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

lang = 'en'
cat = 'adt'
date = '20230105'

def test_string_vio():
    return "Blood war violence"


def run():
    model_name = f'{lang}_{cat}_xlm-multiclass_ow-{date}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None, device=0)  # top_k=None returns all scores; device 0 is for gpus
    for out in classifier(test_string_vio(), padding=True, truncation=True, batch_size=16, num_workers=6, max_length=512)):
        print(out)

run()