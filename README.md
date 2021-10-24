# GANdalf

## Dependencies
torch, torchvision, termcolor

All dependencies should be up-to-date in the Pipfile. Run `pipenv install` to install them to virtual environment, then access the virtual environment with `pipenv shell`.

## Usage
There's an example training file `train.py`. You can run it from the top level of the repo.

Throughout training, example generated images will be saved to `img/[data source]/[algo name]/[number of epochs passed]_epochs.png`. The training example uses MNIST and WGAN-GP and saves examples every epoch, so the 5th file generated would be `img/MNIST/WGAN_GP/5_epochs.png`.

If you have access to a GPU, this library will try to use it the first available one by default.

## Structure
All the important files/folders are listed below.
```
├ gandalf
    ├ algos
        ├ base.py
        ├ gan.py
        ├ wgan_gp.py
        └ wgan.py
    ├ data
        └ mnist.py
    ├ model.py
    └ trainer.py
├ train.py
```

To train a GAN, you have to specify
1. The structure of the generator and discriminator
2. The algorithm you're using to optimize them
3. The data you're training them on

Training happens through the `Trainer` class in `gandalf/trainer.py`. I tried to comment it thoroughly, but feel free to ask me if anything isn't clear. The only real bit of abstraction going on here is the use of the class `Net` from `gandalf/model.py`. It's just a generic neural net wrapper that can take in your generator/discriminator structure and turn it into a bona fide neural net with (hopefully) a simple and intuitive interface.

## Making new algos

Every algo should extend the class `Algo` from `gandalf/algos/base.py`. This class gives you access to a handful of helpful methods that you can access with `self.[method]()`. Every algo needs just 2 methods, `optimize_G` and `optimize_D`. These should take a single optimization step for the generator and discriminator, respectively.

To instantiate a new algo, you should feed it your initial config object. You can then use anything from the config in your algo with `self.[thing you're using]` (whenever you use `self.[something]` and `something` isn't a method in `Algo` or your new algo, it'll search the config for `something`).


<img src="wizard.png">
