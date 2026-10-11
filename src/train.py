import os
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from src.config import (
    MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    MODEL_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS
)
from src.dataset import generate_synthetic_pii_data, tokenize_and_align_labels
from src.evaluator import compute_metrics

def main():
    print("--- Starting Phase 2: Fine-Tuning RoBERTa for PII Detection ---")

    # 1. Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 2. Generate and Split Synthetic Dataset
    print("[1/4] Generating synthetic PII datasets...")
    raw_dataset = generate_synthetic_pii_data(num_samples=1000)
    dataset_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    
    # 3. Tokenize & Align Datasets
    print("[2/4] Tokenizing and aligning sub-word labels...")
    tokenized_datasets = dataset_split.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset_split["train"].column_names
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 4. Configure Training Arguments
    checkpoint_dir = MODEL_DIR / "roberta-pii-checkpoints"
    
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",  # Updated from evaluation_strategy for transformers>=5.0
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none"
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,  # Updated from tokenizer=tokenizer for Transformers 5.x
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 6. Execute Training Loop
    print("[3/4] Training model...")
    trainer.train()

    # 7. Evaluate and Save Best Checkpoint
    print("[4/4] Evaluating and saving best checkpoint...")
    eval_results = trainer.evaluate()
    print(f"\nFinal Evaluation Metrics: {eval_results}\n")

    final_model_path = MODEL_DIR / "best_pii_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Fine-tuned PyTorch model saved successfully to: {final_model_path}")

if __name__ == "__main__":
    main()