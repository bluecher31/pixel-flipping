class ForwardHook:
    "Create a forward hook on module `m` "

    def __init__(self, m, store_output=True):
        self.store_output = store_output
        self.hook = m.register_forward_hook(self.hook_fn)
        self.stored, self.removed = None, False

    def hook_fn(self, module, input, output):
        "stores input/output"
        if self.store_output:
            self.stored = output
        else:
            self.stored = input

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()