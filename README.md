# contextual_loss
contextual loss in TensorFlow2.x/TF2.x/keras

# What's different
[Here](https://github.com/gitlabspy/contextual_loss/blob/3dfbc3b81118408a76ee99f6d599783df3c889c0/contextual_loss/cxloss.py#L49-L51), this is the implementation according to the original repo, however it produces NaN in some of my cases. Therefore I replace it with `tf.math.softmax`.


# usage
```python
import tensorflow as tf
from contextual_loss import contextual_loss 
a = tf.random.normal((2, 64, 64, 256))
b = tf.random.uniform((2, 64, 64, 256), -1.0, 1.0)

print(contextual_loss(a, b, 0.1))
print(contextual_loss(a, a, 0.1))
print(contextual_loss(b, b, 0.1))

```

# Reference
https://github.com/roimehrez/contextualLoss
https://github.com/z-bingo/Contextual-Loss-PyTorch
