from argparse import Action

class ParseStr2List(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)
