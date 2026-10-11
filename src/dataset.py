import random
from faker import Faker
from datasets import Dataset
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH, LABEL2ID

fake = Faker()
Faker.seed(42)
random.seed(42)

def generate_synthetic_pii_data(num_samples: int = 500) -> Dataset:
    """
    Generates synthetic PII text and aligns IOB2 labels for token classification.
    """
    data = []
    
    for _ in range(num_samples):
        tokens = []
        labels = []
        
        # Random template selector
        template_type = random.choice(["email", "api_key", "phone", "clean"])
        
        # 1. Prefix text
        prefix_words = ["Please", "contact", "the", "admin", "at", "system", "user", "key", "is"]
        prefix = random.sample(prefix_words, k=random.randint(2, 4))
        for word in prefix:
            tokens.append(word)
            labels.append(LABEL2ID["O"])
            
        # 2. Inject PII entity or normal text
        if template_type == "email":
            email = fake.email()
            tokens.append(email)
            labels.append(LABEL2ID["B-EMAIL"])
        elif template_type == "api_key":
            key = f"sk-live-{fake.lexify(text='????????????????')}"
            tokens.append(key)
            labels.append(LABEL2ID["B-API_KEY"])
        elif template_type == "phone":
            phone = fake.phone_number()
            phone_tokens = phone.split()
            for idx, p_tok in enumerate(phone_tokens):
                tokens.append(p_tok)
                labels.append(LABEL2ID["B-PHONE"] if idx == 0 else LABEL2ID["I-PHONE"])
        else:
            clean_words = ["service", "running", "normally", "all", "systems", "operational"]
            for word in random.sample(clean_words, k=2):
                tokens.append(word)
                labels.append(LABEL2ID["O"])
                
        # 3. Suffix text
        suffix_words = ["immediately", "for", "support", "verification", "process"]
        suffix = random.sample(suffix_words, k=random.randint(1, 3))
        for word in suffix:
            tokens.append(word)
            labels.append(LABEL2ID["O"])
            
        data.append({"tokens": tokens, "ner_tags": labels})
        
    return Dataset.from_list(data)

def tokenize_and_align_labels(examples, tokenizer):
    """
    Aligns word-level entity tags with sub-word byte-pair encoding (BPE) token splits.
    Special tokens ([CLS], [SEP], padding) are tagged with -100 so loss calculation ignores them.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                # For sub-words of B-TAG, map subsequent pieces to I-TAG or repeat tag
                label_ids.append(labels[word_idx])
            previous_word_idx = word_idx
            
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    raw_dataset = generate_synthetic_pii_data(num_samples=10)
    processed = raw_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    print(f"Phase 1 Verification: Sample batch generated successfully. Total items: {len(processed)}")