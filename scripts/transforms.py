import torch
from torchvision.transforms import functional as TF
import random

def collate_fn(batch):
    return tuple(zip(*batch))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self._last_applied = False
        
    def __call__(self, image, target):
        self._last_applied = (random.random() < self.prob)
        if self._last_applied:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            target = target.flip(-1)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self._last_applied = False
        
    def __call__(self, image, target):
        self._last_applied = (random.random() < self.prob)
        if self._last_applied:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            target = target.flip(-2)
        return image, target
    
class RandomOneOfFlip(object):
    """
    With probability f_a apply exactly one flip.
    If applied: vertical with probability p_v, else horizontal.
    """
    def __init__(self, f_a: float = 0.5, p_v: float = 0.5):
        self.f_a = float(f_a)   # probability that a flip is applied at all
        self.p_v = float(p_v)   # conditional probability of vertical given flip

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        if random.random() < self.f_a:
            if random.random() < self.p_v:
                # print('Vertical')
                image = image.flip(-2)   # vertical
                target = target.flip(-2)
            else:
                # print('Horizontal')
                image = image.flip(-1)   # horizontal
                target = target.flip(-1)
        return image, target

class RandomRightAngleRotate(object):
    """
    Rotate by {0, 90, 180, 270} degrees with equal probability.
    Applies the same rotation to image and target.
    """
    def __init__(self):
        self.k_choices = (0, 1, 2, 3)  # k * 90 degrees

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        k = random.choice(self.k_choices)
        if k != 0:
            # print(k)
            image = torch.rot90(image, k, dims=(-2, -1))
            target = torch.rot90(target, k, dims=(-2, -1))
        return image, target
    
    
class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        target = TF.to_tensor(target)
        return image, target
    
class ToPil(object):
    def __call__(self, image, target):
        image = TF.to_pil_image(image)
        return image, target

    
class Adj_Brightness(object):
    def __init__(self, params, prob = 0.75):
        self.params = params
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            if type(self.params) == tuple:
                transformation_order = random.uniform(self.params[0], self.params[1])
            else:
                transformation_order = random.uniform(0, self.params)
            image = TF.adjust_brightness(image, transformation_order)
        
        return image, target

class Adj_Contrast(object):
    def __init__(self, params, prob = 0.75):
        self.params = params
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            if type(self.params) == tuple:
                transformation_order = random.uniform(self.params[0], self.params[1])
            else:
                transformation_order = random.uniform(0, self.params)
            image = TF.adjust_contrast(image, transformation_order)
            
        return image, target

class Adj_Gamma(object):
    def __init__(self, gain, params, prob = 0.75):
        self.gain = gain
        self.params = params
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            if type(self.params) == tuple:
                transformation_order = random.uniform(self.params[0], self.params[1])
            else:
                transformation_order = random.uniform(0, self.params)
            
            if type(self.gain) == tuple:
                gain_order = random.uniform(self.gain[0], self.gain[1])
            else:
                gain_order = random.uniform(0, self.gain)
            image = TF.adjust_gamma(image, transformation_order, gain_order)
            
        return image, target
    
class Adj_Hue(object):
    def __init__(self, params, prob = 0.75):
        self.params = params
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            if type(self.params) == tuple:
                transformation_order = random.uniform(self.params[0], self.params[1])
            else:
                transformation_order = random.uniform(-0.5, self.params)
            image = TF.adjust_hue(image, transformation_order)
            
        return image, target
    
class Adj_Saturation(object):
    def __init__(self, params, prob = 0.75):
        self.params = params
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            if type(self.params) == tuple:
                transformation_order = random.uniform(self.params[0], self.params[1])
            else:
                transformation_order = random.uniform(-0.5, self.params)
            image = TF.adjust_saturation(image, transformation_order)
            
        return image, target