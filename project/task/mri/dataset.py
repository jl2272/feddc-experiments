from torch.utils.data import Dataset
from pathlib import Path


class MRI(Dataset):
    def __init__(self, path):
        self.paths = []
        for dirname, _, filenames in os.walk(path):
            for filename in sorted(filenames):
                self.paths.append(os.path.join(dirname, filename))
        self.size = len(self.paths)

    def __len__(self):
        return self.size

    def get(self, ix, device, size, pretrained):
        im = cv2.imread(self.paths[ix])[:, :, ::-1]
        im = cv2.resize(im, size)
        im = im / 255.
        im = torch.cuda.FloatTensor(im, device=device)
        im = im.permute(2, 0, 1)
        # if pretrained:
        #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        #    im = normalize(im)
        label = self.paths[ix].split("/")[-2]
        if label == "no":
            target = 0
        if label == "yes":
            target = 1
        return im, label, target

    def __getitem__(self, ix, device, size=(150, 150), pretrained=False):
        im, label, target = self.get(ix, device, size, pretrained)
        return im.to(device).float(), torch.tensor(int(target)).to(device).long()

    def getDataset(self, device, size=(150, 150), pretrained=False):
        imgs = []
        labels = []
        targets = []
        for ix in range(self.size):
            im, label, target = self.get(ix, device, size, pretrained)
            imgs.append(im)
            labels.append(label)
            targets.append(target)
        X = torch.stack(imgs)
        y = torch.tensor(np.array(targets), device=device)
        return X, y



