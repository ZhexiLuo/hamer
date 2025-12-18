from pathlib import Path
import torch
import os
import cv2
import numpy as np
from PIL import Image
import json
import argparse
from flask import Flask, request, jsonify

# --- HaMeR Imports ---
from hamer.models import download_models, load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
import vitpose_model
import hamer

# --- Constants ---
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# --- Hardcoded Paths ---
PROJECT_ROOT = Path(__file__).parent
HAMER_DATA_DIR = PROJECT_ROOT / "_DATA"
HAMER_CHECKPOINT = HAMER_DATA_DIR / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
VITPOSE_SOURCE_DIR = PROJECT_ROOT / "third-party" / "ViTPose"
VITPOSE_DATA_DIR = HAMER_DATA_DIR

# --- Default Config ---
DEFAULT_RESCALE_FACTOR = 2.0
DEFAULT_BATCH_SIZE = 1

class ModelManager:
    """Handles loading of all models and the renderer."""
    def __init__(self):
        print("Initializing ModelManager...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._patch_hamer_paths()

        self.hamer_model, self.model_cfg = self._load_hamer_model()
        self.body_detector = self._load_body_detector()
        self.keypoint_detector = self._load_keypoint_detector()
        self.renderer = Renderer(self.model_cfg, faces=self.hamer_model.mano.faces)

        print(f"Using device: {self.device}")
        print("ModelManager initialized successfully.")

    def _patch_hamer_paths(self):
        import hamer.configs
        hamer.configs.CACHE_DIR_HAMER = str(HAMER_DATA_DIR)

        vitpose_model.ViTPoseModel.MODEL_DICT = {
            'ViTPose+-G (multi-task train, COCO)': {
                'config': str(VITPOSE_SOURCE_DIR / 'configs' / 'wholebody' / '2d_kpt_sview_rgb_img' / 'topdown_heatmap' / 'coco-wholebody' / 'ViTPose_huge_wholebody_256x192.py'),
                'model': str(VITPOSE_DATA_DIR / 'vitpose_ckpts' / 'vitpose+_huge' / 'wholebody.pth'),
            },
        }

    def _load_hamer_model(self):
        download_models(HAMER_DATA_DIR)
        model, model_cfg = load_hamer(str(HAMER_CHECKPOINT))
        model = model.to(self.device)
        model.eval()
        return model, model_cfg

    def _load_body_detector(self):
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "_DATA/detectron2/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)

    def _load_keypoint_detector(self):
        return ViTPoseModel(self.device)

class DetectionManager:
    """Encapsulates hand detection logic."""
    def __init__(self, body_detector, keypoint_detector):
        self.body_detector = body_detector
        self.keypoint_detector = keypoint_detector

    def detect_best_hand(self, img_cv2):
        img = img_cv2.copy()[:, :, ::-1]
        det_out = self.body_detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            return None

        vitposes_out = self.keypoint_detector.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        return self._find_best_hand_from_keypoints(vitposes_out)

    def _find_best_hand_from_keypoints(self, vitposes_out):
        max_confidence = 0
        best_hand_data = None

        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            for is_right, keyp in enumerate([left_hand_keyp, right_hand_keyp]):
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    confidence = keyp[valid, 2].mean()
                    if confidence > max_confidence:
                        max_confidence = confidence
                        bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                        best_hand_data = {
                            "bboxes": np.array([bbox]),
                            "is_right": np.array([is_right]),
                            "keypts_list": np.array([keyp[:, :2]]),
                            "valid_num": np.array([sum(valid)])
                        }
        return best_hand_data

class ReconstructionManager:
    """Handles the core 3D reconstruction."""
    def __init__(self, model, model_cfg, device):
        self.model = model
        self.model_cfg = model_cfg
        self.device = device

    def run(self, img_cv2, hand_data, scaled_focal_length, rescale_factor=DEFAULT_RESCALE_FACTOR):
        dataset = ViTDetDataset(self.model_cfg, img_cv2, hand_data['bboxes'], hand_data['is_right'], rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=0)
        
        batch = next(iter(dataloader))
        batch = recursive_to(batch, self.device)
        with torch.no_grad():
            out = self.model(batch) # pred

        # Process camera parameters
        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center, box_size, img_size = batch["box_center"].float(), batch["box_size"].float(), batch["img_size"].float()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        # Process vertices for single result (assuming batch size 1)
        n = 0
        verts = out['pred_vertices'][n].detach().cpu().numpy()
        is_right_flag = batch['right'][n].cpu().numpy()
        verts[:, 0] = (2 * is_right_flag - 1) * verts[:, 0]
        
        return {
            "vertices": [verts],
            "cam_transl": [pred_cam_t_full[n]],
            "is_right": [is_right_flag],
            "mano_params": {key: v.detach().cpu().numpy() for key, v in out['mano_params'].items()},
            "full_cam_transl": pred_cam_t_full,
            "batch": batch,
            "img_size": img_size,
            "out": out
        }

class OutputManager:
    """Handles saving all output files."""
    def __init__(self, renderer):
        self.renderer = renderer

    def save_all(self, output_dir, img_path, recon_data, hand_data, save_mesh, focal_length, side_view=False, full_frame=True):
        img_fn, _ = os.path.splitext(os.path.basename(str(img_path)))
        person_id = int(recon_data['batch']['personid'][0])

        outputs = {}
        outputs['render_crop'] = self._save_render_crop(output_dir, img_fn, person_id, recon_data, side_view)
        if save_mesh:
            outputs['mesh'] = self._save_mesh(output_dir, img_fn, person_id, recon_data)
        outputs['params_pt'], outputs['params_json'] = self._save_params(output_dir, img_fn, recon_data, hand_data)
        outputs['depth'] = self._save_depth(output_dir, img_fn, recon_data, focal_length, full_frame)
        outputs['camera_params'] = self._save_camera_params(output_dir, img_fn, recon_data, focal_length)

        return outputs

    def _save_render_crop(self, out_folder, img_fn, person_id, recon_data, side_view=False):
        n = 0
        batch = recon_data['batch']
        out = recon_data['out']
        white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
        input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
        input_patch = input_patch.permute(1,2,0).numpy()
        regression_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(), out['pred_cam_t'][n].detach().cpu().numpy(), batch['img'][n], mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1))

        if side_view:
            side_img = self.renderer(out['pred_vertices'][n].detach().cpu().numpy(), out['pred_cam_t'][n].detach().cpu().numpy(), white_img, mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), side_view=True)
            final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
        else:
            final_img = np.concatenate([input_patch, regression_img], axis=1)

        path = os.path.join(str(out_folder), f"{img_fn}_{person_id}.png")
        cv2.imwrite(path, 255 * final_img[:, :, ::-1])
        return path

    def _save_mesh(self, out_folder, img_fn, person_id, recon_data):
        n = 0
        verts = recon_data['vertices'][n]
        cam_t = recon_data['cam_transl'][n]
        is_right = recon_data['is_right'][n]
        tmesh = self.renderer.vertices_to_trimesh(verts, cam_t, LIGHT_BLUE, is_right=is_right)
        path = os.path.join(str(out_folder), f"{img_fn}_{person_id}.obj")
        tmesh.export(path)
        return path

    def _save_params(self, out_folder, img_fn, recon_data, hand_data):
        info = {
            'mano_params': recon_data['mano_params'],
            'cam_transl': recon_data['full_cam_transl'],
            'is_right': recon_data['batch']['right'].cpu().numpy(),
            'valid': hand_data['valid_num'],
            'keypts': hand_data['keypts_list'],
            'boxes': hand_data['bboxes'],
            'batch_size': recon_data['batch']['img'].shape[0]
        }
        pt_path = os.path.join(str(out_folder), f"{img_fn}_params.pt")
        torch.save(info, pt_path)
        json_path = os.path.join(str(out_folder), f"{img_fn}_params.json")
        with open(json_path, 'w') as f:
            json.dump(self._to_serializable(info), f, indent=2, ensure_ascii=False)
        return pt_path, json_path

    def _save_depth(self, out_folder, img_fn, recon_data, focal_length, full_frame=True):
        """
        Save depth map of hand mesh.

        Depth values:
            - Hand region: positive float values (distance from camera in meters)
            - Background (non-hand region): 0.0

        To convert depth to binary mask: mask = (depth > 0)
        """
        if not (full_frame and len(recon_data['vertices']) > 0):
            return None

        n = 0
        misc_args = dict(mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), focal_length=focal_length)
        _, depth = self.renderer.render_rgba_multiple(
            recon_data['vertices'], cam_t=recon_data['cam_transl'],
            render_res=recon_data['img_size'][n], is_right=recon_data['is_right'],
            return_depth=True, **misc_args
        )

        depth_path = os.path.join(str(out_folder), f"{img_fn}_depth.npy")
        np.save(depth_path, depth.astype(np.float32))

        return depth_path

    def _save_camera_params(self, out_folder, img_fn, recon_data, focal_length):
        n = 0
        img_size = recon_data['img_size']

        extrinsics = np.eye(3, 4)
        cam_info = {
            'extrinsics': extrinsics.tolist(),
            'fx': focal_length / img_size[n][0].item(),
            'fy': focal_length / img_size[n][1].item(),
            'cx': 0.5,
            'cy': 0.5
        }

        path = os.path.join(str(out_folder), "camera_params.json")
        with open(path, 'w') as f:
            json.dump(cam_info, f, indent=4)

        return path

    def _to_serializable(self, val):
        # Helper to convert tensors/ndarrays to lists for JSON serialization
        if isinstance(val, np.ndarray): return val.astype(float).tolist()
        if isinstance(val, torch.Tensor): return val.cpu().numpy().astype(float).tolist()
        if isinstance(val, dict): return {k: self._to_serializable(v) for k, v in val.items()}
        if isinstance(val, list): return [self._to_serializable(v) for v in val]
        if isinstance(val, (np.float32, np.float64)): return float(val)
        if isinstance(val, (np.int32, np.int64)): return int(val)
        return val

class HandReconstructionService:
    """Orchestrates the hand reconstruction pipeline."""
    def __init__(self):
        print("Initializing HandReconstructionService...")
        models = ModelManager()
        self.model_cfg = models.model_cfg
        self.detection_manager = DetectionManager(models.body_detector, models.keypoint_detector)
        self.reconstruction_manager = ReconstructionManager(models.hamer_model, models.model_cfg, models.device)
        self.output_manager = OutputManager(models.renderer)
        print("HandReconstructionService initialized successfully.")

    def reconstruct(self, image_path: str, output_dir: str, save_mesh: bool = True,
                    focal_length: float = None, side_view: bool = False,
                    full_frame: bool = True, rescale_factor: float = DEFAULT_RESCALE_FACTOR):
        try:
            os.makedirs(output_dir, exist_ok=True)
            img_cv2 = cv2.imread(str(image_path))
            if img_cv2 is None:
                raise FileNotFoundError(f"Image not found at {image_path}")

            h, w, _ = img_cv2.shape

            # Compute focal length if not provided
            if focal_length is None:
                focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * max(h, w)
            print(f"Image size: {w}x{h}, focal_length={focal_length}")

            hand_data = self.detection_manager.detect_best_hand(img_cv2)
            if hand_data is None:
                return {"status": "warning", "message": "No hands detected."}

            recon_data = self.reconstruction_manager.run(img_cv2, hand_data, focal_length, rescale_factor)
            outputs = self.output_manager.save_all(output_dir, image_path, recon_data, hand_data,
                                                   save_mesh, focal_length, side_view, full_frame)

            return {
                "status": "success",
                "message": "Processing complete.",
                "outputs": outputs
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

# --- Flask App ---
app = Flask(__name__)
hand_reconstruction_service = None

@app.route('/reconstruct', methods=['POST'])
def predict():
    data = request.json
    result = hand_reconstruction_service.reconstruct(
        image_path=data['image_path'],
        output_dir=data['output_dir'],
        save_mesh=data.get('save_mesh', True),
        focal_length=data.get('focal_length'),
        side_view=data.get('side_view', False),
        full_frame=data.get('full_frame', True),
        rescale_factor=data.get('rescale_factor', DEFAULT_RESCALE_FACTOR)
    )
    return jsonify(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HaMeR Hand Reconstruction Server')
    parser.add_argument('--port', type=int, default=5002, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    hand_reconstruction_service = HandReconstructionService()
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)