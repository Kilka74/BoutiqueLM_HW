This Repository contains a code to a model to train it on dataset like the [TinyStories paper](https://arxiv.org/abs/2305.07759). I implement pre-norm Transformer decoder with only Masked Self-Attention, like in original paper, add [Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) and [RMSNorm](https://arxiv.org/pdf/1910.07467.pdf). Also I set gradient accumulation equal to 4. Number of tokens the model was trained on was set to 5 billions.

I have used following hyperparameters:

| batch size                          | 512      |
|-------------------------------------|----------|
| embed dim                           | 512      |
| num heads                           | 8        |
| num layers                          | 8        |
| sequence length                     | 256      |
| tokenization                        | BPE      |
| vocab size                          | 5000     |
| AdamW beta1                         | 0.9      |
| AdamW beta2                         | 0.95     |
| number of params of the final model | 30285824 |