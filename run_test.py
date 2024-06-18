import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import argparse
import time


def main(args):
    def warmup(model):
        conv = get_conversation_template(args.model_type)

        if args.model_type == "llama-2-chat":
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
        elif args.model_type == "mixtral":
            conv = get_conversation_template("llama-2-chat")
            conv.system_message = ''
            conv.sep2 = "</s>"
        conv.append_message(conv.roles[0], "Hello")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if args.model_type == "llama-2-chat":
            prompt += " "
        input_ids = model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()
        for output_ids in model.ea_generate(input_ids):
            ol=output_ids.shape[1]

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.EAGLE_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # set model to eval mode, and warmup the model
    model.eval()
    warmup(model)

    # input message
    your_message="What's the best way to start learning a new language?"

    if args.model_type == "llama-2-chat":
        conv = get_conversation_template("llama-2-chat")  
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " "
    elif args.model_type == "vicuna":
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    input_ids=model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()

    # generate response
    start_time = time.time()
    output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
    end_time = time.time()
    
    # output=model.tokenizer.decode(output_ids[0])
    output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True, spaces_between_special_tokens=False,
                                          clean_up_tokenization_spaces=True, )

    if args.print_time:
        print("Time:", end_time - start_time)

    if args.print_message:
        print("Prompt:")
        print(prompt)
        print("Model response:")
        print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="llama-2-chat",choices=["llama-2-chat","vicuna","mixtral"], help="llama-2-chat or vicuna, for chat template")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The base model path.",
    )
    parser.add_argument(
        "--EAGLE-model-path",
        type=str,
        default="yuhuili/EAGLE-llama2-chat-7B",
        help="The EAGLE model path.",
    )
    parser.add_argument(
        "-pm",
        "--print-message",
        action="store_true",
        help="Print the message.",
    )
    parser.add_argument(
        "-pt",
        "--print-time",
        action="store_true",
        help="Record the time.",
    )
    args = parser.parse_args()
    
    main(args)
    
    