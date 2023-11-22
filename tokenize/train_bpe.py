# THIS CODE IS NOT THE ORIGINAL - IT INCLUDES OUR MODIFICATIONS 

import json
import os
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from datasets import load_dataset

dataset = load_dataset("pubmed") # the dataset will be downloaded and loaded into memory.

input_files = dataset.split(",")

# tokenizer_name = sys.argv[1]
tokenizer_name = "my_name" 

os.system(f"mkdir {tokenizer_name}")

# Initialize a tokenizer
# The use of the Byte Pair Encoding (BPE) tokenizer in the provided script is a specific choice based on the characteristics and advantages of BPE in tokenization tasks.
tokenizer = Tokenizer(models.BPE())


# In the context of the provided code, ByteLevel refers to a specific pre-tokenization, decoding, and post-processing strategy used by the Byte Pair Encoding (BPE) tokenizer provided by the Hugging Face tokenizers library
# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) #The pre-tokenizer is responsible for breaking the input text into smaller units before tokenization. Here, the ByteLevel pre-tokenizer is used, which tokenizes at the byte level. The add_prefix_space=False argument indicates that spaces will not be added before each token, which is useful for languages where spaces are not tokenized.
tokenizer.decoder = decoders.ByteLevel() # The decoder is responsible for converting tokens back into text. The ByteLevel decoder is used, matching the pre-tokenization step 
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True) # The post-processor is applied after tokenization. Here, the ByteLevel post-processor is used with the trim_offsets=True argument, which trims unnecessary offset information.

# And then train

# we could pass show_progress (bool, optional) – Whether to show progress bars while training.... there is many arguments that we can pass to see how the performance changes
trainer = trainers.BpeTrainer(
    vocab_size=28896, # vocab_size: The desired size of the vocabulary, which is set to 28,896.
    min_frequency=2, # min_frequency: The minimum frequency of a token to be included in the vocabulary, set to 2.
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # initial_alphabet: The initial alphabet used for training, which is taken from the ByteLevel pre-tokenizer.
)

# The decoding step occurs after the training of the tokenizer. The training process involves learning the subword units or tokens from the input data, and once the tokenizer is trained, it can be used to tokenize new text. After tokenization, you might want to decode the tokenized sequences back into human-readable text for analysis or presentation.



# the vocabulary is involves extracting and saving information about the vocabulary learned during the training of the Byte Pair Encoding (BPE) tokenizer.
tokenizer.train(input_files,trainer=trainer)

# And Save it
tokenizer.save(f"{tokenizer_name}/tokenizer.json", pretty=True)

# create vocab.json and merges.txt
with open(f"{tokenizer_name}/vocab.json", "w") as vocab_file:
    vocab_json = json.loads(open(f"{tokenizer_name}/tokenizer.json").read())["model"]["vocab"]
    vocab_file.write(json.dumps(vocab_json))

with open(f"{tokenizer_name}/merges.txt", "w") as merges_file:
    merges = "\n".join(json.loads(open(f"{tokenizer_name}/tokenizer.json").read())["model"]["merges"])
    merges_file.write(merges)


