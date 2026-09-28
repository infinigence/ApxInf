"""Exercise the real policy boundary without checkpoint/GPU dependencies."""
import numpy as np
import pytest
import test_qwen_drive_policy as f

config = f.config

@pytest.mark.parametrize('length',[19,3382,3383,3384,3385,3386,3387,3388])
def test_padding_boundary(config,monkeypatch,length):
 p=f.policy(config)
 ids=list(range(1,length+1))
 monkeypatch.setattr(p,'_build_scene_ids',lambda *a:ids)
 seen=[]
 def infer(pixels,grids,tokens,mask,state,embodiment,noise,**options):
  seen.append((tokens.copy(),mask.copy()))
  return np.ones((3,3),np.float32)
 monkeypatch.setattr(p.model_runner,'_infer_preprocessed',infer)
 p.infer(f.scene(),noise=np.zeros((3,3),np.float32))
 tokens,mask=seen[0]
 target=3387 if 3383 <= length <= 3387 else length
 assert len(tokens)==target and len(mask)==target
 np.testing.assert_array_equal(tokens[:length],ids)
 np.testing.assert_array_equal(mask[:length],1)
 np.testing.assert_array_equal(tokens[length:],0)
 np.testing.assert_array_equal(mask[length:],0)
 assert tokens.dtype==np.uint32 and mask.dtype==np.uint8
 assert tokens.flags.c_contiguous and mask.flags.c_contiguous


def test_reasoning_not_padded(config,monkeypatch):
 p=f.policy(config,'reasoning_planning')
 p.infer(f.scene(),noise=np.zeros((3,3),np.float32))
 assert len(p.model_runner.calls[0][2])<3387


def test_native_and_policy_padding_geometry_agree():
 native = pytest.importorskip('apxinf_py')
 assert native.QWEN_DRIVE_FIXED_SCENE_TOKENS == f.qwen_drive.FIXED_SCENE_TOKENS
