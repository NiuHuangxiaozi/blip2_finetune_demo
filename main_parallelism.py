# coding=utf-8

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse
from peft import LoraConfig, get_peft_model
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["pixel_value"], max_length=256, padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["description"]
        return encoding


def plot(loss_list, output_path):
    plt.figure(figsize=(10,5))
    plt.plot(range(len(loss_list)), loss_list, color='#e4007f', label="Train Loss Curve")
    plt.ylabel("Loss", fontsize='large')
    plt.xlabel("Epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')
    plt.savefig(output_path + '/pytorch_image2text_blip2_loss_curve.png')




def main():
    parser = argparse.ArgumentParser(description="BLIP2 Fine-tuning with Accelerate")
    parser.add_argument("--model_id", type=str, default="blip2_2_7b", help="Model ID")
    parser.add_argument("--pretrain_model_path", type=str, default="./models/blip2/", help="Pretrained model path")
    parser.add_argument("--finetune_dataset_path", type=str, default="./dataset/m4_daily_decomposition/trend/", help="Finetune dataset path")
    parser.add_argument("--fineruned_model_output_path", type=str, default="./finetuned_model/", help="Finetuned model output path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    output_path = os.path.join(args.fineruned_model_output_path, args.model_id)
    os.makedirs(output_path, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device

    # Load model and processor
    model = AutoModelForVision2Seq.from_pretrained(args.pretrain_model_path)
    processor = AutoProcessor.from_pretrained(args.pretrain_model_path)

    # Apply LoRA
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    def collator(batch):
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch

    # Load dataset
    dataset = load_from_disk(args.finetune_dataset_path)
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Prepare model, optimizer, and data loader with accelerate
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()
    loss_list = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = []
        
        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss

            epoch_loss.append(loss.item())

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if idx % 10 == 0 and accelerator.is_main_process:
                generated_output = model.generate(pixel_values=pixel_values)
                print(processor.batch_decode(generated_output, skip_special_tokens=True))

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {avg_loss}")
        loss_list.append(avg_loss)

    if accelerator.is_main_process:
        print("Saving model...")
        model.save_pretrained(output_path)
        plot(loss_list, output_path)


if __name__ == "__main__":
    main()
