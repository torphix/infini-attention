# infini-attention

Implementation of the paper: ["Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention
"](https://arxiv.org/abs/2404.07143)

The key idea here is that after calculating the local attention for the current block the attention is cached inside of a long term attention block, this long term attention block is then added to the next sequence processing step.

Note that the caching is done via a special function so that the model chooses what information to store and what information to retrieve from the long term attention store.

Additionally whilst this theoretically allows for storage of long term concepts it seems to me as if it is necessary to include training data samples that have long range dependencies to force the model to recall information from a long time ago other wise it would not have any need to remeber to store long range concepts and thus in inference it would not extract the useful long term information.

Note my implementation has a deviation where i use two seperate gating parameters for local attention and long term attention instead of one like in the paper, I also include a linear projection when querying from the long term memory store.
