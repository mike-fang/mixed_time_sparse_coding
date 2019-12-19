import inspect

class Class:
    def __init__(self, a, b, c=None, d=None):
        self.params = locals() 

c = Class(1, 2, 3, d=4)
print({k:v for (k, v) in c.params.items() if isinstance(v, (int, float, bool)) })
