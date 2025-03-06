# fmt: off
IMAGENET_SUPERCLASS_MAPPING = {
    "random": [100, 200, 300],  # for testing purpose
    "butterfly": [321, 322, 323, 324, 325, 326],
    "boat": [472, 554, 576, 625, 814, 914],
    "car": [407, 436, 468, 511, 609, 627, 656, 661, 751, 817],
    "cat": [281, 282, 283, 284, 285, 286, 287],
    "edible_fruit": [948, 949, 950, 951, 952, 953, 954, 955, 956, 957],
    "fungus": [991, 993, 994, 995, 996, 997],
    "truck": [ 555, 569, 656, 675, 717, 734, 864, 867],
}
# fmt: on


from . import original, subclasses, some_vs_others
