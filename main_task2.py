from huggingface_hub import login
import os, datetime, torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from Gemma3_utiles import (
    process_vision_info_inference,
    process_text_info,
    find_mask_idx,
    decode_for_verification,
)
from compute_loss import task2_compute_metric
import pdb

# =========[ 基础设置 ]=========
os.environ["WANDB_MODE"] = "disabled"
hf_token = "yourtoken"
login(hf_token)

dummy_train = Dataset.from_dict({"dummy": [0]}) #dummy train dataset for pass the SFTtrainer init
eval_metadata_path = "./task2_test_data.jsonl"
eval_dataset = load_dataset("json", data_files=eval_metadata_path)["train"]

print(f"✅ Validation dataset length: {len(eval_dataset)}")

USE_QLORA = True
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config if USE_QLORA else None,
)

model_task2 = AutoModelForImageTextToText.from_pretrained("./checkpoints/checkpoint-task2", **model_kwargs)
processor = AutoProcessor.from_pretrained("./checkpoints/checkpoint-task2")

def custom_collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info_inference(example["messages"])
        text_str = process_text_info(example["messages"])
        texts.append(text_str.strip())
        images.append(image_inputs)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    none_masked_phrase_idx, none_masked_special_idx = find_mask_idx(batch, processor)
    labels = batch["input_ids"].clone()
    labels_masked = torch.full_like(batch["input_ids"], -100)
    for i in range(len(none_masked_special_idx)):
        for j in range(len(none_masked_special_idx[i])):
            start, end = none_masked_special_idx[i][j]
            labels_masked[i, start:end] = labels[i, start:end]
    batch["labels"] = labels_masked
    return batch

args = SFTConfig(
    output_dir="./eval_results",
    per_device_eval_batch_size=1,
    bf16=True,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    prediction_loss_only=False,
    batch_eval_metrics = True,
)

trainer = SFTTrainer(
    model=model_task2,
    args=args,
    train_dataset=dummy_train, #dummy train dataset for pass the SFTtrainer init
    eval_dataset=eval_dataset,
    processing_class=processor,
    data_collator=custom_collate_fn,
    compute_metrics=task2_compute_metric,
)

print("🚀 Start evaluating model...")
metrics = trainer.evaluate()

print("✅ Evaluation completed.")
print(metrics)
