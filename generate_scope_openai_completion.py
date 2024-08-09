import torch
from openai import OpenAI
import huggingface_hub
from transformers import AutoTokenizer, AutoModel
from fairseq.models.roberta import RobertaModel

from tqdm import tqdm
import argparse
import logging

HF_TOKEN = "YOUR_HF_TOKEN"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

huggingface_hub.login(token=HF_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)
device = "cuda:0"

# openai.util.logger.setLevel(logging.ERROR)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def scope_generate(
    input_txt,
    client,
    tokenizer_gpt,
    editor,
    tokenizer,
    gen_len,
    b_size,
    model_id="davinci-002",
    iter_num=1,
    seed=0,
):
    torch.manual_seed(seed)
    b_num = int(gen_len // b_size)

    if gen_len % b_size != 0:
        print("Block number is not integer")
        return

    input_editor = tokenizer.encode(input_txt)[:-1]
    len0 = len(input_editor) - 1
    while len(tokenizer.encode(input_txt)[1:-1]) < (gen_len + len0):
        n_gen_toks = 0
        generated_encoded = []

        while n_gen_toks <= b_size:
            prompt = f"{input_txt}"

            response = client.completions.create(
                model=model_id, prompt=prompt, max_tokens=50
            )

            generated_string = response.choices[0].text

            if len(input_txt) > 0 and input_txt in generated_string:
                generated_string = input_txt.join(generated_string.split(input_txt)[1:])

            generated_encoded += tokenizer.encode(generated_string)[1:-1]
            n_gen_toks += len(generated_encoded)
            input_txt += generated_string

        input_editor += generated_encoded[: b_size + 1]
        inputs_editor = torch.tensor(
            [input_editor + [2]], dtype=torch.long, device=device
        )

        for _ in range(iter_num):
            logits = editor(inputs_editor)[0][:, -b_size - 2 :, :]
            sampler = torch.distributions.categorical.Categorical(logits=logits)
            edited = sampler.sample()[:, :]
            edited[:, -2:] = inputs_editor[:, -2:].clone()
            edited = torch.concat([inputs_editor[:, : -b_size - 2], edited], dim=-1)
            inputs_editor = edited
        edited_string = tokenizer.decode(
            edited[0, 1:-2], clean_up_tokenization_spaces=False
        )

        input_txt = edited_string
        input_editor = inputs_editor[0, :-2].tolist()

    return input_txt, inputs_editor[:, 1:-2]


parser = argparse.ArgumentParser()
parser.add_argument("--target_domain", type=str, default="camera")
parser.add_argument("--source_domain", type=str, default="camera")
parser.add_argument(
    "--scope_path", type=str, default="checkpoints/scope_camera_review_gpt2xl"
)
parser.add_argument("--input_txt", type=str, default="This text is about")
parser.add_argument("--model_id", type=str, default="davinci-002")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

target_domain = args.target_domain
source_domain = args.source_domain
scope_path = args.scope_path
input_txt = args.input_txt
model_id = args.model_id
seed = args.seed

tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding_side="left")

editor = RobertaModel.from_pretrained(scope_path, "checkpoint_last.pt")
editor = editor.model.half()
editor.eval()
editor = editor.to(device)

scope_gen_str, scope_gen_tok = scope_generate(
    input_txt,
    client,
    tokenizer_gpt,
    editor,
    tokenizer,
    128,
    16,
    model_id=model_id,
    iter_num=5,
    seed=seed,
)

print(scope_gen_str)
