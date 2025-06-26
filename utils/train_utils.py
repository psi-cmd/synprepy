import signal
from contextlib import contextmanager

@contextmanager
def sigint_ignored():
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        print('Now ignoring CTRL-C')
        yield
    except:
        raise  # Exception is dropped if we don't reraise it.
    finally:
        print('Returning control to default signal handler')
        signal.signal(signal.SIGINT, original_sigint_handler)