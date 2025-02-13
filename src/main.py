import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

from watermarking import generate, detect_watermark, get_perplexities

class GPT2Wrapper(torch.nn.Module):
    def __init__(self):
        super(GPT2Wrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs.logits


def main():
    set_seed(0)
    
    device = Accelerator().device

    model = GPT2Wrapper().to(device)
    vocab_size = model.tokenizer.vocab_size

    prior = model.tokenizer("Some text to be continued",
                            return_tensors="pt")["input_ids"].to(device)

    normal_ids = generate(model, prior, max_length=200, watermark=False)
    n_ppl = get_perplexities(model, normal_ids).item()    
    n_z = detect_watermark(normal_ids, vocab_size).item()  

    watermarked_ids = generate(model, prior, max_length=200, watermark=True)
    w_ppl = get_perplexities(model, watermarked_ids).item()   
    w_z = detect_watermark(watermarked_ids, vocab_size).item() 

    print(
        f"\n\n\033[92mNormal text (PPL = {n_ppl:.2f}, Z-statistic = {n_z:.2f})\033[0m:\n")
    print(model.tokenizer.decode(normal_ids[0]))

    print(
        f"\n\n\033[93mWM text (PPL = {w_ppl:.2f}, Z-statistic = {w_z:.2f})\033[0m:\n")
    print(model.tokenizer.decode(watermarked_ids[0]))


if __name__ == "__main__":
    main()
