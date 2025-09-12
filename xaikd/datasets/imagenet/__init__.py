# fmt: off
IMAGENET_SUPERCLASS_MAPPING = {
    "random": [100, 200, 300],  # for testing purpose
    "obsolte-butterfly": [321, 322, 323, 324, 325, 326],
    "obsolte-boat": [472, 554, 576, 625, 814, 914],
    "obsolte-car": [407, 436, 468, 511, 609, 627, 656, 661, 751, 817],
    "obsolte-cat": [281, 282, 283, 284, 285, 286, 287],
    "obsolte-edible_fruit": [948, 949, 950, 951, 952, 953, 954, 955, 956, 957],
    "obsolte-fungus": [991, 993, 994, 995, 996, 997],
    "obsolte-truck": [ 555, 569, 656, 675, 717, 734, 864, 867],

    # we have 8 superclasses containing 10 classes
    # [81] wading bird, wader
    "wading-bird": [129, 130, 134, 135, 138],
    # [115] retriever
    "retriever": [205, 206, 207, 208, 209],
    # [121] working dog
    "working-dog": [242, 243, 246, 247, 248],
    # [137] domestic cat, house cat, Felis domesticus, Felis catus
    "domestic-cat": [281, 282, 283, 284, 285],
    # [202] bag
    "bag": [414, 636, 728, 748, 797],
    # [217] bottle
    "bottle": [440, 720, 737, 898, 907],
    # [221] box
    "box": [478, 492, 519, 637, 709],
    # [482] truck, motortruck
    "truck": [555, 569, 717, 864, 867],
}
# fmt: on


from . import original, subclasses, some_vs_others
