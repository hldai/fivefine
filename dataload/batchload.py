
def train_batch_list_iter(train_batch_loader, gacc_step):
    batch_list = list()
    for batch in train_batch_loader:
        batch_list.append(batch)
        if len(batch_list) == gacc_step:
            yield batch_list
            batch_list = list()


class IterExampleBatchLoader:
    def __init__(self, example_loader, batch_size, n_iter=-1, n_steps=-1, collect_fn=None):
        self.example_loader = example_loader
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_steps = n_steps
        self.collect_fn = collect_fn

    def __iter__(self):
        return self.next_batch()

    def next_batch(self):
        it = 0
        step = 0
        batch_examples = list()
        while it != self.n_iter and step != self.n_steps:
            example_iter = iter(self.example_loader)
            for i, example in enumerate(example_iter):
                batch_examples.append(example)
                if len(batch_examples) >= self.batch_size:
                    if self.collect_fn is not None:
                        yield self.collect_fn(batch_examples)
                    else:
                        yield batch_examples
                    batch_examples = list()

                    step += 1
                    if step == self.n_steps:
                        break
            if len(batch_examples) > 0:
                if self.collect_fn is not None:
                    yield self.collect_fn(batch_examples)
                else:
                    yield batch_examples
                batch_examples = list()
            it += 1
