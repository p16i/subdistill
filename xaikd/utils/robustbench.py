try:
    from robustbench.data import load_cifar100c
except ImportError as e:
    # this catches `autoattack` import missing

    if "autoattack" in str(e):
        print("[warning] autoattack is missing")
    else:
        raise e

from robustbench.data import load_cifar100c
from robustbench.utils import load_model
