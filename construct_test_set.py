import torch
from tqdm import tqdm

blocked_file_test = open("./amazon-raw/camera/camera.test.gpt_block", "r")
lines_test = blocked_file_test.readlines()

f = open("./amazon-raw/camera/camera.test_32_128", "w")

encoded_whole = []
for line in lines_test:
    encoded = list(map(int, line.strip().split(" ")))
    encoded_whole += encoded

toks_num = len(encoded_whole)
samp_num = 0
torch.manual_seed(1)
with tqdm(total=toks_num) as pbar:
    while len(encoded_whole) >= 160:
        samp = encoded_whole[:160]
        samp_str = " ".join(list(map(str, samp))) + "\n"
        write_switch = (torch.rand([1]) < 0.4).item()
        if write_switch:
            f.write(samp_str)
            samp_num += 1
        encoded_whole = encoded_whole[160:]
        pbar.update(160)

print("Wrote", samp_num, "samples")
f.close()
