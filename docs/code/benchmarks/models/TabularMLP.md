# TabularMLP.cpp

Two-layer classifier for a feature table: `FullyConnected`, `LeakyReLU(0.1)`, `FullyConnected`. `forward_logits` walks that list. `get_all_layers` returns it, so the trainer and `Network::save` see every weight.

The constructor rejects a non-positive feature count, hidden size, or class count.
