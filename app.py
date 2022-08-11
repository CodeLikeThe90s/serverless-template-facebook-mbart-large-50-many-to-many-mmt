from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    global torch_device
    
    model_name = 'facebook/mbart-large-50-many-to-many-mmt'
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch Device:{}".format(torch_device))
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(torch_device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer
    global torch_device

    input = model_inputs.get('input', None)
    src_lang = model_inputs.get('src_lang', None)
    tgt_lang = model_inputs.get('tgt_lang', None)
    min_length = model_inputs.get('min_length', None)
    max_length = model_inputs.get('max_length', None)
    max_new_tokens = model_inputs.get('max_new_tokens', None)
    if input  == None:
        return {'message': "No input provided"}
    if src_lang == None:
        return {'message': "No src_lang provided"}
    if tgt_lang == None:
        return {'message': "No tgt_lang provided"}
    
    # Run the model
    tokenizer.src_lang = src_lang
    encoded_src = tokenizer(input, return_tensors="pt").to(torch_device)
    generated_tokens = model.generate(
        **encoded_src,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length = max_length,
        min_length= min_length,
        max_new_tokens=max_new_tokens
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # Return the results as a dictionary
    return result
