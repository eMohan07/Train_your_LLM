import torch


class ShakespeareDataset:

    def __init__(self, tokenizer, train_split=0.9):

        self.data = torch.tensor(
            tokenizer.encode(tokenizer.text),
            dtype=torch.long
        )

        n = int(train_split * len(self.data))

        self.train_data = self.data[:n]

        self.val_data = self.data[n:]

    def get_batch(
        self,
        split,
        batch_size,
        context_length,
        device,
    ):

        data = (
            self.train_data
            if split == "train"
            else self.val_data
        )

        ix = torch.randint(
            len(data) - context_length,
            (batch_size,)
        )

        x = torch.stack([
            data[i:i + context_length]
            for i in ix
        ])

        y = torch.stack([
            data[i + 1:i + context_length + 1]
            for i in ix
        ])

        return x.to(device), y.to(device)