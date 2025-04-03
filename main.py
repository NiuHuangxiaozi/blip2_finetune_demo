
# coding=utf-8

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
import os
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["pixel_value"], max_length=256, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["description"]
        return encoding


def plot(loss_list, output_path):
    plt.figure(figsize=(10,5))

    freqs = [i for i in range(len(loss_list))]
    # 绘制训练损失变化曲线
    plt.plot(freqs, loss_list, color='#e4007f', label="image2text train/loss curve")

    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.savefig(output_path+'/pytorch_image2text_blip2_loss_curve.png')
    # plt.show()


def main():

    print(f"exp begin!")
    # load args
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--model_id", type=str, default="blip2_2_7b", 
                        help="model uuid")
    parser.add_argument("--pretrain_model_path", required=False, 
                        type=str, default="./models/blip2/", help="pretrain model's path")
    parser.add_argument("--finetune_dataset_path", type=str,
                        default="./dataset/m4_daily_decomposition/trend/", 
                        help="finetune dataset path")
    parser.add_argument("--fineruned_model_output_path", type=str,
                        default="./finetuned_model/",
                        help="finetuned model output path")
    parser.add_argument("--epoches", type=int,
                        default=3,
                        help="finetuned epoch")
    parser.add_argument("--use_cpu", type=bool,
                        default=False,
                        help="debug on cpu.")
    args = parser.parse_args()
    pprint(vars(args),indent = 4)
    
    pretrain_model_path = args.pretrain_model_path
    finetune_dataset_path = args.finetune_dataset_path
    fineruned_model_output_path = args.fineruned_model_output_path+args.model_id+'/'
    
    if not os.path.exists(fineruned_model_output_path):
        os.makedirs(fineruned_model_output_path)
    

    # We load our model and processor using `transformers`
    if args.use_cpu:
         model = AutoModelForVision2Seq.from_pretrained(pretrain_model_path,  device_map='cpu')
    else:    
        model = AutoModelForVision2Seq.from_pretrained(pretrain_model_path, load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(pretrain_model_path)
    
    # Let's define the LoraConfig
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    # Get our peft model and print the number of trainable parameters
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Let's load the dataset here!
    # 如何将多个数据组合为一个batchsize
    def collator(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        
        # list all keys
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
    
    
    dataset = load_from_disk(finetune_dataset_path)
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=8,
                                  collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device =  'cuda'

    model.train()
    loss_list = []
    for epoch in range(args.epoches):
        print("Epoch:", epoch)
        sum_loss_list = []
        for idx, batch in tqdm(enumerate(train_dataloader)):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

            loss = outputs.loss

            print("Loss:", loss.item())

            sum_loss_list.append(float(loss.item()))
                
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if idx % 10 == 0:
                generated_output = model.generate(pixel_values=pixel_values)
                print(processor.batch_decode(generated_output, skip_special_tokens=True))
            
        avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
        print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
        loss_list.append(float(avg_sum_loss))

    print("model_output:", fineruned_model_output_path)
    model.save_pretrained(fineruned_model_output_path)
    plot(loss_list, '.')



if __name__ == "__main__":
    main()
