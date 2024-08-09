import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import math

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_ID"
huggingface_hub.login(token=HF_TOKEN)
device = "cuda:0"

tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
tokenizer_gpt.pad_token_id = tokenizer_gpt.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    "gpt2-xl", torch_dtype=torch.float16, device_map="balanced"
)
model.config.pad_token_id = model.config.eos_token_id

with open("./amazon-raw/camera/amazon_camera.raw", "r") as f:
    amazon_raw = f.readlines()

# amazon_us_reviews dataset is defunct and no longer accessible due to the decision of data providers.
# amazon_raw = load_dataset("amazon_us_reviews", "Camera_v1_00")

splits = ["train", "valid", "test"]

encoded = {}
for split in splits:
    encoded[split] = []
lines_1d_train = ""
lines_1d_valid = ""
lines_1d_test = ""
for i, line in enumerate(tqdm(amazon_raw)):
    if len(line) < 1:
        continue
    if i % 50 == 0:
        lines_1d_valid += line + "\n\n\n"
    elif i % 50 == 1:
        lines_1d_test += line + "\n\n\n"
    else:
        lines_1d_train += line + "\n\n\n"
    if i % 1000 == 0:
        encoded["train"] += tokenizer_gpt.encode(lines_1d_train)
        encoded["valid"] += tokenizer_gpt.encode(lines_1d_valid)
        encoded["test"] += tokenizer_gpt.encode(lines_1d_test)
        lines_1d_train = ""
        lines_1d_valid = ""
        lines_1d_test = ""
print(
    f"Encoded {len(encoded['train'])} tokens for train, {len(encoded['valid'])} for validation, {len(encoded['test'])} for test"
)

print("Chunking tokens for RoBERTa fine-tuning...")
for split in splits:
    with open(f"./amazon-raw/camera/camera.{split}.gpt_block", "w") as f:
        block_size = 510
        max_pt = len(encoded[split])
        ptr = 0
        sample_n = 0
        with tqdm(total=max_pt) as pbar:
            while ptr < max_pt:
                ptr_next = min(ptr + block_size, max_pt)
                block = encoded[split][ptr:ptr_next]
                samp_str = " ".join(list(map(str, block))) + "\n"
                f.write(samp_str)
                ptr = ptr_next
                pbar.update((ptr_next - ptr))
                sample_n += 1
    print(f"Processed {sample_n} chunks for {split} in RoBERTa fine-tuning")

print("Start construction of dataset for ScoPE training...")
for split in splits:
    with open(f"./amazon-raw/camera/camera.{split}.gpt2xl", "w") as f:
        ptr = 0
        ptr1 = 0
        ptr2 = 0
        bs = 200
        prefix_lens = torch.arange(129)[32:]
        lens_num = len(prefix_lens)
        prefix_lens = prefix_lens.repeat(1, math.ceil(bs / lens_num) + 1).squeeze()
        max_pt = len(encoded[split])
        sample_num = 0
        seed = 1
        torch.manual_seed(seed)

        with tqdm(total=max_pt) as pbar:
            while ptr < max_pt:
                prefix_l = []
                s = ptr
                l = prefix_lens[s % lens_num : s % lens_num + bs]
                b = torch.randint(2, 34, (bs,))
                l_cumsum = l.cumsum(dim=0)
                bs_prac = (l_cumsum + s <= max_pt).int().sum().item()
                if bs_prac == 0:
                    ptr += l_cumsum[0]
                    continue
                l_prac = l[:bs_prac]
                l_prac_cumsum = l_prac.cumsum(dim=0)
                batch_chunk = torch.tensor(
                    encoded[split][s : s + l_prac_cumsum[-1]]
                ).cuda()
                batch = (
                    torch.ones(
                        [l_prac.size(0), l_prac.max()], dtype=torch.int64, device=device
                    )
                    * tokenizer_gpt.pad_token_id
                )
                chunk_st_idx = 0
                for i in range(l_prac.size(0)):
                    st_idx = batch.size(1) - l_prac[i]
                    batch[i, st_idx:] = batch_chunk[chunk_st_idx : l_prac_cumsum[i]]
                    chunk_st_idx = l_prac_cumsum[i]

                prefix_l += l_prac.tolist()
                ptr += l_prac_cumsum[-1].item()
                sample_num += batch.size(0)
                attn_mask = (batch != tokenizer_gpt.pad_token_id).type(torch.int64)

                outputs = model.generate(
                    batch,
                    attention_mask=attn_mask,
                    do_sample=True,
                    max_length=batch.size(1) + b.max().item(),
                    min_length=batch.size(1) + b.max().item(),
                )
                outputs = outputs.cpu()

                for j in range(outputs.size(0)):
                    sample = outputs[j]
                    samp_filt = list(filter((50256).__ne__, sample.tolist()))
                    samp_filt = samp_filt[: (prefix_l[j] + b[j].item())]
                    sample_str = (
                        " ".join(list(map(str, samp_filt)))
                        + " "
                        + str(b[j].item())
                        + "\n"
                    )
                    f.write(sample_str)

                pbar.update(l_prac_cumsum[-1].item())
        print(f"Generated {sample_num} samples for {split}")
