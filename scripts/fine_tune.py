# scripts/fine_tune.py

import argparse
import logging
import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on a domain-specific dataset"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/domain_dataset.txt",
        help="Path to domain-specific text dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fine_tuned_model",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--block_size", type=int, default=128, help="Block size for text chunking"
    )
    args = parser.parse_args()

    # Load the dataset from a text file
    logger.info(f"Loading dataset from {args.data_file}")
    raw_datasets = load_dataset("text", data_files={"train": args.data_file})
    
    # Split the dataset into training and validation (90% train, 10% eval)
    logger.info("Splitting dataset into train and validation sets")
    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    # Load the GPT-2 tokenizer and set the pad token
    logger.info("Loading GPT-2 tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a pad token by default

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    logger.info("Tokenizing training and validation datasets")
    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Function to group texts into blocks of block_size
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // args.block_size) * args.block_size
        result = {}
        for k, t in concatenated.items():
            result[k] = [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Grouping tokenized texts into blocks")
    tokenized_train = tokenized_train.map(group_texts, batched=True)
    tokenized_eval = tokenized_eval.map(group_texts, batched=True)

    # Load pre-trained GPT-2 model and resize token embeddings
    logger.info("Loading pre-trained GPT-2 model")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
    )

    # Initialize Trainer
    logger.info("Starting training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()

    # Save the fine-tuned model and tokenizer
    logger.info(f"Saving the fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate the model and calculate perplexity
    logger.info("Evaluating the model")
    eval_results = trainer.evaluate()
    loss = eval_results["eval_loss"]
    perplexity = torch.exp(torch.tensor(loss))
    logger.info(f"Perplexity: {perplexity.item()}")


if __name__ == "__main__":
    main()
