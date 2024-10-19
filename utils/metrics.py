import torch

def predict_on_sample(model,tokenizer,max_seq_len, train_ds, id_to_class, data,device, n):
    sampled_data = data.sample(n)
    prompts = sampled_data['prompt'].tolist()
    actual_labels = sampled_data['label_str']

    model.eval()
    with torch.no_grad():
        for prompt, act_label in zip(prompts, actual_labels):
            tokenized_prompt = tokenizer.encode(prompt).ids
            required_pads = max_seq_len - len(tokenized_prompt) +1
            padded_tokens = torch.cat(
                [
                torch.tensor(tokenized_prompt,dtype=torch.int64),
                torch.tensor([train_ds.pad_token_id] * required_pads,dtype=torch.int64)
                ]
                )
            padded_tokens = torch.tensor(padded_tokens,dtype=torch.int64).to(device)
            model_output = model.encode(padded_tokens)
            predicted_class_id = torch.argmax(model_output, dim=-1).item()
            print("prompt :", prompt)
            print("act_label :", act_label)
            print("predicted :", id_to_class[predicted_class_id])
            print()