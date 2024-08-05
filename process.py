import os, glob, cv2, argparse, torch, rembg
import numpy as np
from PIL import Image
from seem.masks import FG_remove


class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=256, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")
    # SEEM
    parser.add_argument('--seem_ckpt', type=str, default="../Tools/SEEM/focall_unicl_lang_demo.yaml", help='restore where the SEEM & LaMa model locates')
    parser.add_argument('--seem_cfg', type=str, default="../Tools/SEEM/seem_focall_v0.pt")
    # LaMa
    parser.add_argument('--lama_ckpt', default='../Tools/LaMa/', help='actually path to lama ckpt base folder, ckpt specified in config files')
    parser.add_argument('--lama_cfg', default='./configs/lama_default.yaml', help='path to lama inpainting config path')

    #Outputs
    parser.add_argument('--results', default='../v2x-outputs/pre-process/', help='result direction')


    opt = parser.parse_args()

    """
        create results directions here ↓
        MASK: os.path.join(opt.results, 'masks')
        VIEW: os.path.join(opt.results,  'views')
    """
    if not os.path.exists(opt.results):
        os.mkdir(opt.results)
    if not os.path.exists(os.path.join(opt.results, 'remove')):
        os.mkdir(os.path.join(opt.results, 'remove'))


    setattr(opt, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # session = rembg.new_session(model_name=opt.model)

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*')
        out_dir = opt.path
    else: # isfile
        files = [opt.path]
        out_dir = os.path.dirname(opt.path)


    
    for file in files:

        out_base = os.path.basename(file).split('.')[0]
        out_rgba = os.path.join(out_dir, out_base + '_rgba.png')

        # load image
        print(f'[INFO] loading image {file}...')
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED) # read
        
        # TODO: use seem to remove foreground
        print(f'[INFO] background removal...')

        carved_image, mask = FG_remove(opt = opt, img = image)
        # TODO: save intermediate results
        cv2.imwrite(os.path.join(opt.results, 'remove/r.jpg'), cv2.cvtColor(np.uint8(carved_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(opt.results, 'remove/m.jpg'), cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2BGR))


    print('\nDone.')
