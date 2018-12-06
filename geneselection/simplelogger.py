class SimpleLogger:
    def __init__(self, print_format):

        self.print_format = print_format

        self.log = dict()

    def add(self, input):
        if not self.log:
            for k in input:
                self.log[k] = [input[k]]
        else:
            for k in input:
                self.log[k].append(input[k])

        if self.print_format is not None:
            print(self.print_format.format(**input))

    def __len__(self):

        values = self.log.values()
        if not len(values):
            return 0

        return len(next(iter(values)))
