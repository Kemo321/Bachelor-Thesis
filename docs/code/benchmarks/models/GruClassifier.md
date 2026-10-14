# GruClassifier.cpp

Sequence classifier. The first layer is `Gru` (`[N, T, F]` to the final hidden state `[N, H]`). The second is `FullyConnected` onto the classes. `forward_logits` binds the cuDNN stream and walks that list.

The constructor rejects a non-positive input size, hidden size, or class count.
