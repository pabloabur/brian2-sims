class ParamDict(dict):
    """
    This is just a convenient class used to generate an error if the
    provided key is not present on the original dictionary. This prevents
    undeclared entries to go unnoticed so it is easier to notice when a
    non-existent connection is being handled.
    """
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f'You tried to change {key}, which is has not been '
                           f'previously initilized on the model. You are only '
                           f'supposed to change existent parameters')
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for key, val in dict(*args, **kwargs).items():
            if key not in self:
                raise KeyError(f'You tried to change {key}, which is has not been '
                               f'previously initilized on the model. You are only '
                               f'supposed to change existent parameters')
            self[k] = v

class BaseEquation:
    def __init__(self):
        self.model = ''
        self.namespace = ParamDict({})
        self.parameters = ParamDict({})

    def modify_model(self, attr, new_expr, old_expr=None, key=None):
        """
        Modify model.

        Parameters
        ----------
        attr : str
            Indicates which key of the model's dictionary is to be modified.
        new_expr : str
            New expression that will replace old values.
        old_expr : str
            Specifies which part of the old expression is to be replaced. If
            not provided, entire expression is used.
        key : str
            Key of the dictionary identified by parameter attr.

        Notes
        -----
        The model itself is a class with attributes that can be str or dict. In
        the latter case, using the parameter key is necessary. Using strings for
        values in dictionary namespace will raise an error, so use explicit
        values instead.

        Examples
        --------
        >>> model.modify_model('threshold', 'Vm >= 10')
        >>> model.modify_model('threshold', ' < ', old_expr=' >= ')
        >>> model.modify_model('namespace', 0, old_expr=127, key='Vthr')
        """
        desc = getattr(self, attr)
        if isinstance(desc, str):
            if key:
                raise ValueError(f'Argument key={key}, but it should be '
                                 f'different from None only when object '
                                 f'to be changed is a dictionary. Did you '
                                 f'mean old_expr?')

            if old_expr:
                if old_expr not in desc:
                    raise ValueError(f'{old_expr} is not present on '
                                     f'model\'s attr. Ensure you only change '
                                     f'existing expression')
                desc = desc.replace(old_expr, new_expr)
            else:
                desc = new_expr
            setattr(self, attr, desc)

        elif isinstance(desc, dict):
            desc[key] = new_expr

    def print_model(self):
        for var in vars(self):
            print('\n' + var + ':\n============')
            var_desc = getattr(self, var)
            try:
                var_desc = [x.strip() for x in var_desc.split('\n')]
                for v in var_desc:
                    if not v:
                        continue
                    print(v)
            except AttributeError:
                # Prints dict ATM
                print(var_desc)

class BaseSynapse(BaseEquation):
    def __init__(self):
        super().__init__()
        self.on_pre = ''
        self.on_post = ''
        self.on_event = 'spike'
        self.connection = {'condition': None, 'i': None, 'j': None,
                           'p': 1.}

class BaseNeuron(BaseEquation):
    def __init__(self):
        super().__init__()
        self.threshold = ''
        self.refractory = ''
        self.reset = ''
        self.events = None
