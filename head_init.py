"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Mapping from CIFAR100 to ImageNet labels
_CIFAR100_TO_IMAGENET = {
    "apple": ["Granny Smith", "custard apple"],
    "aquarium_fish": ["goldfish", "anemone fish", "rock beauty"],
    "baby": ["cradle", "crib", "bassinet"],
    "bear": ["brown bear", "American black bear", "ice bear", "sloth bear"],
    "beaver": ["beaver"],
    "bed": ["four-poster", "crib", "bassinet"],
    "bee": ["bee", "apiary", "honeycomb"],
    "beetle": [
        "tiger beetle",
        "ladybug",
        "ground beetle",
        "long-horned beetle",
        "leaf beetle",
        "dung beetle",
        "rhinoceros beetle",
        "weevil",
    ],
    "bicycle": ["mountain bike", "bicycle-built-for-two"],
    "bottle": [
        "water bottle",
        "pop bottle",
        "beer bottle",
        "pill bottle",
        "wine bottle",
    ],
    "bowl": ["mixing bowl", "soup bowl"],
    "boy": ["ballplayer", "groom"],
    "bridge": ["suspension bridge", "steel arch bridge", "viaduct"],
    "bus": ["school bus", "minibus", "trolleybus"],
    "butterfly": ["monarch", "ringlet", "cabbage butterfly", "sulphur butterfly", "lycaenid"],
    "camel": ["Arabian camel"],
    "can": ["milk can", "can opener"],
    "castle": ["castle", "palace"],
    "caterpillar": ["walking stick", "centipede", "grasshopper"],
    "cattle": ["ox", "water buffalo", "bison"],
    "chair": ["folding chair", "rocking chair", "barber chair", "throne"],
    "chimpanzee": ["chimpanzee"],
    "clock": ["analog clock", "digital clock", "wall clock"],
    "cloud": ["bubble", "geyser"],
    "cockroach": ["cockroach"],
    "couch": ["studio couch"],
    "crab": ["Dungeness crab", "rock crab", "fiddler crab", "king crab", "hermit crab"],
    "crocodile": ["African crocodile", "American alligator"],
    "cup": ["cup", "coffee mug", "measuring cup", "goblet"],
    "dinosaur": ["triceratops"],
    "dolphin": ["killer whale", "dugong", "sea lion"],
    "elephant": ["African elephant", "Indian elephant", "tusker"],
    "flatfish": ["tench", "coho", "sturgeon", "gar"],
    "forest": ["valley", "lakeside", "promontory"],
    "fox": ["red fox", "kit fox", "Arctic fox", "grey fox"],
    "girl": ["gown", "maillot"],
    "hamster": ["hamster"],
    "house": ["mobile home", "boathouse", "birdhouse", "church", "monastery"],
    "kangaroo": ["wallaby"],
    "keyboard": ["computer keyboard", "typewriter keyboard"],
    "lamp": ["table lamp", "lampshade"],
    "lawn_mower": ["lawn mower"],
    "leopard": ["leopard", "snow leopard"],
    "lion": ["lion"],
    "lizard": [
        "banded gecko",
        "common iguana",
        "American chameleon",
        "whiptail",
        "agama",
        "frilled lizard",
        "alligator lizard",
        "Gila monster",
        "green lizard",
        "African chameleon",
    ],
    "lobster": ["American lobster", "spiny lobster"],
    "man": ["groom", "ballplayer"],
    "maple_tree": ["buckeye", "acorn"],
    "motorcycle": ["motor scooter", "moped"],
    "mountain": ["alp", "cliff", "promontory", "volcano", "valley"],
    "mouse": ["mouse"],
    "mushroom": [
        "mushroom",
        "coral fungus",
        "agaric",
        "gyromitra",
        "stinkhorn",
        "earthstar",
        "hen-of-the-woods",
        "bolete",
    ],
    "oak_tree": ["acorn"],
    "orange": ["orange"],
    "orchid": ["yellow lady's slipper"],
    "otter": ["otter"],
    "palm_tree": ["banana", "pineapple"],
    "pear": ["Granny Smith", "custard apple", "fig"],
    "pickup_truck": ["pickup"],
    "pine_tree": ["acorn", "buckeye", "lumbermill"],
    "plain": ["airliner", "warplane", "wing", "plane"],
    "plate": ["plate", "plate rack"],
    "poppy": ["rapeseed", "daisy"],
    "porcupine": ["porcupine"],
    "possum": ["mink", "weasel", "armadillo"],
    "rabbit": ["wood rabbit", "hare", "Angora"],
    "raccoon": ["skunk", "badger", "polecat"],
    "ray": ["electric ray", "stingray"],
    "road": ["street sign", "traffic light", "parking meter"],
    "rocket": ["missile", "projectile", "space shuttle"],
    "rose": ["hip", "vase"],
    "sea": ["seashore", "sandbar", "lakeside", "coral reef"],
    "seal": ["sea lion"],
    "shark": ["great white shark", "tiger shark", "hammerhead"],
    "shrew": ["mouse", "weasel", "mink"],
    "skunk": ["skunk"],
    "skyscraper": ["obelisk", "dome"],
    "snail": ["snail"],
    "snake": [
        "thunder snake",
        "ringneck snake",
        "hognose snake",
        "green snake",
        "king snake",
        "garter snake",
        "water snake",
        "vine snake",
        "night snake",
        "boa constrictor",
        "rock python",
        "Indian cobra",
        "green mamba",
        "sea snake",
        "horned viper",
        "diamondback",
        "sidewinder",
    ],
    "spider": [
        "black and gold garden spider",
        "barn spider",
        "garden spider",
        "black widow",
        "tarantula",
        "wolf spider",
        "spider web",
    ],
    "squirrel": ["fox squirrel"],
    "streetcar": ["streetcar", "trolleybus"],
    "sunflower": ["daisy", "rapeseed"],
    "sweet_pepper": ["bell pepper"],
    "table": ["dining table", "desk", "pool table"],
    "tank": ["tank", "half track"],
    "telephone": ["cellular telephone", "dial telephone", "pay-phone"],
    "television": ["television", "monitor", "screen", "home theater"],
    "tiger": ["tiger"],
    "tractor": ["tractor"],
    "train": ["steam locomotive", "electric locomotive", "bullet train", "passenger car"],
    "trout": ["tench", "coho", "sturgeon", "gar"],
    "tulip": ["vase", "daisy", "yellow lady's slipper"],
    "turtle": ["loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle"],
    "wardrobe": ["wardrobe", "chiffonier"],
    "whale": ["grey whale", "killer whale"],
    "willow_tree": ["lakeside", "valley"],
    "wolf": ["timber wolf", "white wolf", "red wolf", "coyote"],
    "woman": ["gown", "maillot", "kimono"],
    "worm": ["flatworm", "nematode"],
}

def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------

    # Load old weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    old_model = models.resnet18(weights=weights)
    old_weights = old_model.fc.weight.detach()
    old_bias = old_model.fc.bias.detach()
    imagenet_cats = weights.meta['categories']
    imagenet_cats_to_idx = {name: idx for idx, name in enumerate(imagenet_cats)}

    new_weights = []
    new_bias = []
    for cifar_class in _CIFAR100_TO_IMAGENET.keys():
        imagenet_ids = [imagenet_cats_to_idx[name] for name in _CIFAR100_TO_IMAGENET[cifar_class]]

        new_weights.append(old_weights[imagenet_ids].mean(dim=0))
        new_bias.append(old_weights[imagenet_ids].mean())
    new_weights = torch.stack(new_weights).to(layer.weight.dtype)
    new_bias = torch.stack(new_bias).to(layer.bias.dtype)

    with torch.no_grad():
        layer.weight.copy_(new_weights)
        layer.bias.copy_(new_bias)

    # -------------------------------------------------------------------------
