import os
import sys
import glob
import numpy as np
import PIL.Image

celeba_dir = '/research/hal-datastage/datasets/processed/CelebA'

def rot90(v):
    return np.array([-v[1], v[0]])

class CreateCeleba:
    def __init__(self):
        print('Loading CelebA from "%s"' % celeba_dir)
        self.expected_images = 202599
        if len(glob.glob(os.path.join(celeba_dir, 'img_celeba', '*.jpg'))) != self.expected_images:
            error('Expected to find %d images' % self.expected_images)
        with open(os.path.join(celeba_dir, 'Anno', 'list_landmarks_celeba.txt'), 'rt') as file:
            landmarks = [[float(value) for value in line.split()[1:]] for line in file.readlines()[2:]]
            self.landmarks = np.float32(landmarks).reshape(-1, 5, 2)
        self.size = 256
        self.save_path = os.path.join(celeba_dir, 'celebahq_crop', 'imgs')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def process_func(self, idx):
        # Load original image.
        orig_file = '{:06d}.jpg'.format(idx+1)
        orig_path = os.path.join(celeba_dir, 'img_celeba', orig_file)
        save_path = os.path.join(self.save_path, orig_file)
        if os.path.exists(save_path):
            return
        img = PIL.Image.open(orig_path)

        # Choose oriented crop rectangle.
        lm = self.landmarks[idx]
        eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
        mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
        eye_to_eye = lm[1] - lm[0]
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = self.size / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, PIL.Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(self.size * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(self.size * 0.3 / zoom)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
            quad += pad[0:2]

        # Transform.
        img = img.transform((4*self.size, 4*self.size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        img = img.resize((self.size, self.size), PIL.Image.ANTIALIAS)
        img.save(save_path)
        print(orig_file)

celeba_creator = CreateCeleba()
for i in range(0, celeba_creator.expected_images):
    celeba_creator.process_func(i)
