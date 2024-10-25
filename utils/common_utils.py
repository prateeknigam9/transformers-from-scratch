import torch    

def save_checkpoints(state:dict, filename = "checkpoint.pth.tar"):
    print("==> Saving model checkpoint")
    torch.save(state, filename)

def load_checkpoints(checkpoint,model,optimizer):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond,mask=True)
        
        logits = logits[:, -1, :]  
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def check_model_post_training(tokenizer, model, max_token_len=6, seq_len=256):
    while True:
        txt_input = input("Enter starting words (or type 'exit' to quit): ")
        
        if txt_input.strip().lower() == 'exit':
            print("Exiting...")
            return True
        
        if not txt_input.strip():
            print("Please enter some text.")
            continue
        
        encoded_txt = torch.tensor(tokenizer.encode(txt_input)).unsqueeze(0)
        out = generate_text_simple(
            model=model,
            idx=encoded_txt, 
            max_new_tokens=max_token_len, 
            context_size=seq_len
        )

        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        print(decoded_text)