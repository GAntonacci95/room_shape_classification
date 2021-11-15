from datetime import datetime
import threading


class ThreadTimer:
    def __init__(self):
        self.__first = None
        self.__beginning = None
        self.__end_first = None
        self.__end_total = None
        return

    def start(self):
        if self.__first is None:
            self.__beginning = datetime.now()
            print("Thread({}) - Start({})".format(
                threading.current_thread().ident,
                self.__beginning
            ))
            self.__first = True
        return

    @staticmethod
    def elapsed_s(end: datetime, start: datetime):
        return (end - start).total_seconds()

    def end_first(self, n_rpt: int):
        if self.__first:
            self.__end_first = datetime.now()
            elapsed = self.elapsed_s(self.__end_first, self.__beginning)
            print("Thread({}) - {:.3f}[\"] x {} -> Forecast({}['])".format(
                threading.current_thread().ident,
                elapsed,
                n_rpt,
                round(elapsed * n_rpt / 60)
            ))
            self.__first = False
        return

    def end_total(self):
        self.__end_total = datetime.now()
        print("Thread({}) - End({}) - Actual({}['])".format(
            threading.current_thread().ident,
            self.__end_total,
            round(self.elapsed_s(self.__end_total, self.__beginning) / 60)
        ))
        return
