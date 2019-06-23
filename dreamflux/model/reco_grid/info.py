class Info(object):
    fields = 'time', 'loss', 'acc'

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
        self.loss = None
        self.acc = None

    def dump(self):
        return self.__dict__

    def to_text(self):
        lines = [
            '* Time     %5.1fms' % (1000 * self.time,),
            '* Loss     %7.3f' % self.loss,
            '* Accuracy %6.2f%%' % (100 * self.acc,),
        ]
        return ''.join(map(lambda line: line + '\n', lines))
