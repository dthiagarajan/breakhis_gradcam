import importlib, sys, types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub modules required by breakhis_gradcam.data
np_stub = types.ModuleType('numpy')
np_stub.random = SimpleNamespace(permutation=lambda x: list(range(len(x))), beta=lambda a,b,size: [0]*size)
np_stub.random.seed = lambda x: None
np_stub.concatenate = lambda arr, axis=0: arr[0]
np_stub.geomspace = lambda start, stop, num: [start]*(num)
sys.modules.setdefault('numpy', np_stub)

class DummyTensor(list):
    def long(self):
        return self
    def squeeze(self):
        return self

torch_stub = types.ModuleType('torch')
torch_stub.Tensor = DummyTensor
torch_stub.manual_seed = lambda x: None
torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
torch_stub.utils = SimpleNamespace(data=SimpleNamespace(Dataset=object))
sys.modules.setdefault('torch', torch_stub)

# torchvision transforms stub
tv_stub = types.ModuleType('torchvision')
tv_transforms_stub = types.ModuleType('torchvision.transforms')
setattr(tv_stub, 'transforms', tv_transforms_stub)
sys.modules.setdefault('torchvision', tv_stub)
sys.modules.setdefault('torchvision.transforms', tv_transforms_stub)

# PIL.Image stub
pil_stub = types.ModuleType('PIL')
image_stub = types.ModuleType('PIL.Image')
image_stub.open = lambda fp: f"image:{fp}"
pil_stub.Image = image_stub
sys.modules.setdefault('PIL', pil_stub)
sys.modules.setdefault('PIL.Image', image_stub)

# Now import the module under test
data_module = importlib.import_module('breakhis_gradcam.data')


def test_getitem_without_transform(tmp_path):
    img = tmp_path / 'dummy.jpg'
    img.write_text('x')
    ds = data_module.BreaKHisDataset([((str(img), 't','s','m','s','q'), 1)], transform=None)
    item, label = ds[0]
    assert item == f"image:{img}"
    assert isinstance(label, data_module.torch.Tensor)
    assert label[0] == 1


def test_getitem_with_transform(tmp_path):
    img = tmp_path / 'dummy.jpg'
    img.write_text('x')
    ds = data_module.BreaKHisDataset([((str(img), 't','s','m','s','q'), 2)], transform=lambda x: f"processed:{x}")
    item, label = ds[0]
    assert item == f"processed:image:{img}"
    assert label[0] == 2
