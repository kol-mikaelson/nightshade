import argparse
import glob
import os
import pickle

from PIL import Image
from torchvision import transforms

from opt import PoisonGeneration


def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


def main(directory, target_name, outdir, eps):
    poison_generator = PoisonGeneration(target_concept=target_name, device="cuda", eps=eps)
    all_data_paths = glob.glob(os.path.join(directory, "*.p"))
    all_imgs = [pickle.load(open(f, "rb"))['img'] for f in all_data_paths]
    all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    all_imgs = [Image.fromarray(img) for img in all_imgs]

    all_result_imgs = poison_generator.generate_all(all_imgs, target_name)
    os.makedirs(outdir, exist_ok=True)

    for idx, cur_img in enumerate(all_result_imgs):
        cur_data = {"text": all_texts[idx], "img": cur_img}
        pickle.dump(cur_data, open(os.path.join(outdir, "{}.p".format(idx)), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str,
                        help="", default='')
    parser.add_argument('-od', '--outdir', type=str,
                        help="", default='')
    parser.add_argument('-e', '--eps', type=float, default=0.04)
    parser.add_argument('-t', '--target_name', type=str, default="cat")
    return parser.parse_args(argv)


if __name__ == '__main__':
    ddir = r"C:\Users\aero7\PycharmProjects\nightshade-release-main\outdata"
    target = "cat"
    outdir = r"C:\Users\aero7\PycharmProjects\nightshade-release-main\outdata2"
    eps = 0.04
    main(ddir, target, outdir, eps)
