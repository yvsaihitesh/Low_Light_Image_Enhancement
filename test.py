import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from net.CIDNet import CIDNet


def parse_args():
    parser = argparse.ArgumentParser(description='Test CIDNet (No Tiling, High Quality)')

    parser.add_argument('--input_dir',  type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/test/')
    parser.add_argument('--weights', type=str, required=True)

    parser.add_argument('--max_offset', type=float, default=3.0)

    parser.add_argument('--cpu', action='store_true', default=False)

    return parser.parse_args()



def pad_to_multiple(img, multiple=32):
    _, _, H, W = img.shape

    new_H = (H + multiple - 1) // multiple * multiple
    new_W = (W + multiple - 1) // multiple * multiple

    pad_H = new_H - H
    pad_W = new_W - W

    img = F.pad(img, (0, pad_W, 0, pad_H), mode='reflect')
    return img, H, W



IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def get_image_paths(input_dir):
    return [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]


def load_image(path, device):
    img = Image.open(path).convert('RGB')
    tensor = transforms.ToTensor()(img)
    return tensor.unsqueeze(0).to(device)


def save_output(tensor, save_path):
    tensor = tensor.clamp(0.0, 1.0).cpu()
    save_image(tensor, save_path)



def main():
    args = parse_args()

    
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('Running on', device)

    
    print(f'Loading weights: {args.weights}')
    model = CIDNet(max_offset=args.max_offset).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = get_image_paths(args.input_dir)
    if len(image_paths) == 0:
        print('No images found')
        return

    print(f'Found {len(image_paths)} images\n')

    for idx, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        name, _ = os.path.splitext(fname)
        out_path = os.path.join(args.output_dir, f'{name}_enhanced.png')

        print(f'[{idx+1}/{len(image_paths)}] {fname}', end=' ... ')

        try:
            img = load_image(img_path, device)

            
            img, orig_H, orig_W = pad_to_multiple(img, 32)

            with torch.no_grad():
                output = model(img)

            
            output = output[:, :, :orig_H, :orig_W]

            save_output(output, out_path)
            print(f'saved')

        except Exception as e:
            print(f'\nERROR: {e}')
            raise

    print(f'\nDone! Saved to {args.output_dir}')


if __name__ == '__main__':
    main()