from transformers import AutoTokenizer
import torch
import mauve
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target_domain", type=str, default="camera")
parser.add_argument("--source_domain", type=str, default="camera")
args = parser.parse_args()
target_domain = args.target_domain
source_domain = args.source_domain

tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

path_gold = (
    "./amazon-raw/" + target_domain + "/" + target_domain.split("_")[0] + ".test_32_128"
)
with open(path_gold, "r") as f_gold:
    lines_gold = f_gold.readlines()
lines_gold = lines_gold[:2000]
p_text = []
for i in range(2000):
    tokens_p = torch.tensor(list(map(int, lines_gold[i].strip().split(" "))))
    str_p = tokenizer_gpt.decode(tokens_p)
    p_text.append(str_p)

domains = [source_domain]
keys = ["scope_iter5"]
for domain in domains:
    print(domain + " prefix")
    lines = {}
    mauve_outs = {}
    # path_test = './amazon-raw/' + target_domain + '/' + domain + '.test_32_128.' + plm + '.' + key
    path_test = (
        f"amazon-raw/{target_domain}/{domain}.test_32_128.davinci-002.scope_iter5"
    )
    f_test = open(path_test, "r")
    with open(path_test, "r") as f_test:
        lines_test = f_test.readlines()
    lines_test = lines_test[:2000]
    lines[keys[0]] = lines_test
    mauve_outs[keys[0]] = torch.tensor([])

    for seed in range(5):
        for key in keys:
            q_text = []
            for i in range(2000):
                tokens_q = torch.tensor(
                    list(map(int, lines[key][i].strip().split(" ")))
                )
                str_q = tokenizer_gpt.decode(tokens_q)
                q_text.append(str_q)
            mauve_out = mauve.compute_mauve(
                p_text=p_text,
                q_text=q_text,
                device_id=0,
                max_text_length=160,
                verbose=False,
                seed=seed,
                batch_size=100,
            )
            mauve_outs[key] = torch.concat(
                [mauve_outs[key], torch.tensor([mauve_out.mauve])], dim=0
            )
    for key in keys:
        s = key + " :"
        print(s, end=" ")
        for seed in range(5):
            s = str(round(mauve_outs[key][seed].item(), 4))
            print(s, end=" ")
        s = "| avg : " + str(round(mauve_outs[key].mean().item(), 4))
        print(s, end=" ")
        s = "| std : " + str(round(mauve_outs[key].std().item(), 4))
        print(s)
