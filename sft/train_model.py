import argparse
import logging
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Script for Partial Fine-Tuning a model on SFT dataset")
    parser.add_argument("--model_id", type=str, default="google/gemma-2-2b-it", help="HuggingFace Model ID")
    parser.add_argument("--dataset", type=Path, default=Path("sft/sft_dataset.jsonl"), help="Path to JSONL dataset")
    parser.add_argument("--output_dir", type=Path, default=Path("sft/tuned_model"), help="Output directory for fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of final transformer layers to unfreeze")
    
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error(f"Dataset {args.dataset} not found. Generate it first using create_sft_dataset.py.")
        return

    logger.info(f"=== STEP 1: Loading Dataset from {args.dataset} ===")
    dataset = load_dataset("json", data_files={"train": str(args.dataset)})

    logger.info(f"=== STEP 2: Loading Model and Tokenizer ({args.model_id}) ===")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
    )

    logger.info(f"=== STEP 3: Freezing weights / Partial Fine-tuning setup ===")
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        logger.warning("Could not automatically locate transformer layers. Will just unfreeze lm_head.")
        layers = []

    if len(layers) > 0:
        num_layers_to_unfreeze = args.unfreeze_layers
        unfrozen_start = max(0, len(layers) - num_layers_to_unfreeze)
        logger.info(f"Unfreezing last {num_layers_to_unfreeze} layers: from idx {unfrozen_start} to {len(layers)}")
        for layer in layers[unfrozen_start:]:
            for param in layer.parameters():
                param.requires_grad = True

    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,d} || all params: {all_params:,d} || "
        f"trainable%: {100 * trainable_params / all_params:.4f}%"
    )

    logger.info("=== STEP 4: Setting up SFT ===")
    
    response_template = "<start_of_turn>model\n"
    
    try:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, 
            tokenizer=tokenizer
        )
    except Exception as e:
        logger.warning(f"Could not init DataCollatorForCompletionOnlyLM: {e}. Falling back to default.")
        data_collator = None

    logger.info("=== STEP 5: Setting up Trainer ===")
    training_args = TrainingArguments(
        output_dir=str(Path("sft/checkpoints")),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        max_seq_length=2048,
    )

    logger.info("=== STEP 6: Starting Training ===")
    model.config.use_cache = False
    trainer.train()

    logger.info(f"=== STEP 7: Saving Model to {args.output_dir} ===")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    
    logger.info("Done! Fine-tuned model is saved and ready.")

if __name__ == "__main__":
    main()
