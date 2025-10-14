    if "image" in sample and "images" not in sample:
        sample["images"] = sample["image"]
    if "images" in sample:
        images = []
        # images.append(Image.open(BytesIO(sample["images"])).convert("RGB"))
        if isinstance(sample["images"], str):
            images.append(Image.open(sample["images"]).convert("RGB"))
        elif isinstance(sample["images"], list):
            images.append(Image.open(sample["images"][0]).convert("RGB"))
        else:
            images.append(Image.open(BytesIO(sample["images"])).convert("RGB"))




@dataclass
class MyDataArguments(DataArguments):
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )

@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)



    transform = partial(
        process_sample,
        processor=processor,
        chat_template=chat_template,
        position_id_func=position_id_func,
        source_name=args.data.source_name,
    )



def sharegpt4v_pretrain_preprocess_customer(conversations, generation_ratio=0.0, **kwargs):
    """
    row data: {"from": [role], "value": [content]}
    transform data: [{"from": role, "value": content}, {"from": role, "value": content}, ...]
    """
    
    if isinstance(conversations, dict) and "from" in conversations and "value" in conversations:
        if len(conversations["from"]) == len(conversations["value"]):
            conversations = [
                {"from": f, "value": v} 
                for f, v in zip(conversations["from"], conversations["value"])
            ]
        else:
            raise ValueError("Length of 'from' and 'value' lists must be equal")
    
    constructed_conversation = []
    
    if conversations and conversations[0]["from"] != "human":
        conversations = conversations[1:]
    
    if not conversations:
        raise ValueError("Empty conversations after preprocessing")
    assert conversations[0]["from"] == "human", "First message must be from human"

    for message in conversations:
        role = message["from"]
        value = message["value"]
        if role == "human":
            value = value.replace("<image>", "")
            constructed_conversation.append(["user", ("image", None), ("text", value)])
        else:
            if value is not None:
                constructed_conversation.append(["assistant", ("text", value)])
            else:
                constructed_conversation.append(None)

    generate_sample = random.random() < generation_ratio
    if generate_sample:
        caption = constructed_conversation[-1][1][1] if constructed_conversation[-1] else ""
        instruction = f"Generate an image based on the following caption: {caption}"
        constructed_conversation = [["user", ("text", instruction)], ["assistant", ("image", None)]]
    
    return constructed_conversation



DATASETS = {
    "sharegpt4v_pretrain": sharegpt4v_pretrain_preprocess,
    "sharegpt4v_pretrain_customer": sharegpt4v_pretrain_preprocess_customer,


QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "flash_attention_2": Qwen2_5_VLSdpaAttention,
    "sdpa": Qwen2_5_VLSdpaAttention,
}







