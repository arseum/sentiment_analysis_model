import itertools
import sys
import threading
import time


def _in_jupyter():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def run_with_spinner(fn, msg="Traitement en cours"):
    """Exécute fn() avec un spinner animé.

    - En terminal  : spinner Braille animé sur la même ligne via \\r
    - En Jupyter   : affichage IPython avec clear_output pour l'animation
    """
    if _in_jupyter():
        return _run_jupyter(fn, msg)
    return _run_terminal(fn, msg)


def _run_terminal(fn, msg):
    frames = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    done = threading.Event()

    def _spin():
        for frame in frames:
            if done.is_set():
                break
            sys.stdout.write(f'\r{frame}  {msg}')
            sys.stdout.flush()
            time.sleep(0.1)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        done.set()
        t.join()
        sys.stdout.write('\r' + ' ' * (len(msg) + 5) + '\r')
        sys.stdout.flush()
    return result


def _run_jupyter(fn, msg):
    from IPython.display import clear_output, display
    frames = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    done = threading.Event()

    def _spin():
        for frame in frames:
            if done.is_set():
                break
            clear_output(wait=True)
            print(f'{frame}  {msg}', flush=True)
            time.sleep(0.15)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        done.set()
        t.join()
        clear_output(wait=True)
    return result
