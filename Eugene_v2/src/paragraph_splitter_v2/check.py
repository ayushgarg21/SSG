from transformers import AutoConfig

config = AutoConfig.from_pretrained("mirth/chonky_modernbert_base_1")
print(config.architectures)
print(config.problem_type)