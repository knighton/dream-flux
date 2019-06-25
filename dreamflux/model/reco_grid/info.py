class Info(object):
    fields = 'time', 'forward_time', 'backward_time', 'input_time', 'core_time', 'core_spread_time', 'core_remix_time', 'output_time', 'loss', 'acc'

    @classmethod
    def mean(cls, infos):
        ret = Info()

        for field in cls.fields:
            setattr(ret, field, [])

        for info in infos:
            for field in cls.fields:
                value = getattr(info, field)
                getattr(ret, field).append(value)

        for field in cls.fields:
            values = getattr(ret, field)
            setattr(ret, field, sum(values) / len(values))

        return ret

    def __init__(self):
        self.time = None
        self.forward_time = None
        self.backward_time = None
        self.input_time = None
        self.core_time = None
        self.core_spread_time = None
        self.core_remix_time = None
        self.output_time = None
        self.loss = None
        self.acc = None

    def dump(self):
        return self.__dict__

    def to_text(self):
        lines = [
            '* Time        %5.1fms' % (1000 * self.time,),
            '  * Forawrd   %6.1fms' % (1000 * self.forward_time,),
            '  * Backard   %6.1fms' % (1000 * self.backward_time,),
            '  * Input     %6.1fms' % (1000 * self.input_time,),
            '  * Core      %6.1fms' % (1000 * self.core_time,),
            '    * Spread  %6.1fms' % (1000 * self.core_spread_time,),
            '    * Remix   %6.1fms' % (1000 * self.core_remix_time,),
            '  * Output    %6.1fms' % (1000 * self.output_time,),
            '* Loss        %7.3f' % self.loss,
            '* Accuracy    %6.2f%%' % (100 * self.acc,),
        ]
        return ''.join(map(lambda line: line + '\n', lines))
