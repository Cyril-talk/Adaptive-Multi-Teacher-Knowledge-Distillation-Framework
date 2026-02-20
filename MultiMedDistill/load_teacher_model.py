import sys 
sys.path.append('./BiomedParse/')
sys.path.append('./MedSAM/')
sys.path.append('./RETFound_MAE/')
sys.path.append('./usfm/')
import warnings 
warnings.filterwarnings('ignore')
import torch
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class MedSAMWrapper:
    """优化的MedSAM模型包装器，处理输入格式转换"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()  # 默认设为评估模式
        self.transform = ResizeLongestSide(1024)  # MedSAM的标准输入尺寸
        self.device = next(model.parameters()).device  # 获取模型设备
        
    def __call__(self, input_list, multimask_output=False):
        """处理前向传播调用"""
        # 确保模型处于评估模式
        self.model.eval()
        
        try:
            # 预处理输入图像
            batched_input = self._preprocess_batch(input_list)
            
            # 调用原始模型
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(batched_input, multimask_output)
            
            # 提取logits并转换为单精度
            if isinstance(outputs, list):
                # 提取MedSAM的输出logits并转换为单精度
                logits_list = []
                for output in outputs:
                    if 'low_res_logits' in output:
                        logits_list.append(output['low_res_logits'])
                    elif 'masks' in output:
                        logits_list.append(output['masks'])
                
                if logits_list:
                    teacher_output = torch.cat(logits_list).float()
                else:
                    return None
            else:
                if hasattr(outputs, 'low_res_logits'):
                    teacher_output = outputs.low_res_logits.float()
                elif hasattr(outputs, 'masks'):
                    teacher_output = outputs.masks.float()
                else:
                    teacher_output = outputs.float()
                    
            return teacher_output
            
        except Exception as e:
            print(f"Warning: MedSAM forward error: {e}")
            return None
    
    def extract_features(self, input_list):
        """提取特征 - 只返回图像编码器的输出"""
        # 确保模型处于评估模式
        self.model.eval()
        
        try:
            # 预处理输入图像并获取张量
            input_tensor = self._preprocess_images(input_list)
            
            # 使用半精度提取图像特征
            with torch.no_grad(), torch.cuda.amp.autocast():
                features = self.model.image_encoder(input_tensor)
                
            # 转换为单精度
            return features.float()
        except Exception as e:
            print(f"Warning: MedSAM feature extraction error: {e}")
            return None
    
    def _preprocess_batch(self, input_list):
        """预处理整个批次为MedSAM期望的输入格式"""
        batched_input = []
        for item in input_list:
            image = item["image"]
            # 应用MedSAM的预处理
            transformed = self.transform.apply_image(image)
            image_tensor = torch.as_tensor(transformed, device=self.device)
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()
            
            # 归一化
            pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device)
            pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device)
            image_tensor = (image_tensor - pixel_mean[:, None, None]) / pixel_std[:, None, None]
            
            # 创建MedSAM期望的输入格式
            batched_input.append({
                "image": image_tensor,
                "original_size": image.shape[:2]  # 保存原始尺寸
            })
        return batched_input
    
    def _preprocess_images(self, input_list):
        """仅预处理图像为张量，不创建字典列表"""
        images = []
        for item in input_list:
            image = item["image"]
            # 应用MedSAM的预处理
            transformed = self.transform.apply_image(image)
            image_tensor = torch.as_tensor(transformed, device=self.device)
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()
            
            # 归一化
            pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device)
            pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device)
            image_tensor = (image_tensor - pixel_mean[:, None, None]) / pixel_std[:, None, None]
            
            images.append(image_tensor)
        
        return torch.stack(images)
    
    def to(self, device):
        self.model.to(device)
        self.device = device  # 更新设备信息
        return self
    
    def eval(self):
        """设置模型为评估模式"""
        self.model.eval()
        return self

def load_biomedprase_model():
    try:
        from BiomedParse.modeling.BaseModel import BaseModel
        from BiomedParse.modeling import build_model
        from BiomedParse.utilities.distributed import init_distributed
        from BiomedParse.utilities.arguments import load_opt_from_config_files
        from BiomedParse.utilities.constants import BIOMED_CLASSES

        opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
        opt = init_distributed(opt)
        # Load model from pretrained weights
        pretrained_pth = '/root/autodl-tmp/teacher_model/biomedparse_v1.pt'

        model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

        return model
    except Exception as e:
        print(f"Error loading BioMedPrase model: {e}")
        return None

def load_MedSAM_model():
    try:
        from MedSAM.segment_anything import sam_model_registry
        MedSAM_CKPT_PATH = "/root/autodl-tmp/teacher_model/medsam_vit_b.pth"
        device = "cuda:0"
        
        medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH)
        medsam_model = MedSAMWrapper(medsam_model)
        medsam_model = medsam_model.to(device)
        
        return medsam_model
    except Exception as e:
        print(f"Error loading MedSAM model: {e}")
        return None

def load_USFM_model():
    try:
        import torch
        from omegaconf import OmegaConf
        from usfm.models.beitSegLit import BeitSegLit
        cfg_file = "/root/projects/upernet_beit_base.yaml"
        net_cfg = OmegaConf.load(cfg_file)['net']
        teacher = BeitSegLit(net=net_cfg,
                            optimizer=None,
                            scheduler=None,
                            metric_keys=[])
        model_path = '/root/autodl-tmp/teacher_model/USFM_latest.pth'
        state = torch.load(model_path, map_location='cuda:0', weights_only=False)
        teacher.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state,
                                strict=False)
        teacher = teacher.eval().cuda()
        return teacher
    except Exception as e:
        print(f"Error loading USFM model: {e}")
        return None

def load_RETFound_MAE_model():
    try:
        import torch
        import RETFound_MAE.models_mae as models_mae
        arch = 'mae_vit_large_patch16'
        model = getattr(models_mae, arch)()
        chkpt_dir = '/root/autodl-tmp/teacher_model/RETFound_oct_weights.pth'
        # load model
        checkpoint = torch.load(chkpt_dir, weights_only=False)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        model = model.eval().cuda()
        return model
    except Exception as e:
        print(f"Error loading RETFound_MAE model: {e}")
        return None

def load_models():
    model_dict = {}
    
    # print("Loading teacher models...")
    
    biomed_model = load_biomedprase_model()
    if biomed_model is not None:
        model_dict["BioMedPrase"] = biomed_model
        print("✓ BioMedPrase loaded")
    else:
        print("✗ BioMedPrase load failed")
    
    medsam_model = load_MedSAM_model()
    if medsam_model is not None:
        model_dict["MedSAM"] = medsam_model
        print("✓ MedSAM loaded")
    else:
        print("✗ MedSAM load failed")
    
    retfound_model = load_RETFound_MAE_model()
    if retfound_model is not None:
        model_dict["RETFound_MAE"] = retfound_model
        print("✓ RETFound_MAE loaded")
    else:
        print("✗ RETFound_MAE load failed")
    
    usfm_model = load_USFM_model()
    if usfm_model is not None:
        model_dict["USFM"] = usfm_model
        print("✓ USFM loaded")
    else:
        print("✗ USFM load failed")
    
    # print(f"Successfully loaded {len(model_dict)} teacher models: ")
    
    return model_dict

if __name__ == "__main__":
    teacher_models = load_models()    
    print(f"Loaded models: {list(teacher_models.keys())}")
