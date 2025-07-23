import os
import time


os.environ["PYGLET_HEADLESS"] = "True"

from multiprocessing.pool import Pool
from multiprocessing import get_context, TimeoutError,Process
from functools import partial

import numpy as np
from tqdm import tqdm


class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc

def init_worker():
    # runs exactly once per worker, before any tasks

    os.environ["OMP_NUM_THREADS"]       = "1"
    os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    os.environ["MKL_NUM_THREADS"]       = "1"
    
    import trimesh
    from scipy.spatial import cKDTree

    from normal_consistency import compute_normals_metrics
    import cadquery as cq

    # make them available to your metric code
    globals()['trimesh'] = trimesh
    globals()['cKDTree'] = cKDTree
    globals()['compute_normals_metrics'] = compute_normals_metrics
    globals()['cq'] = cq



    
def init_worker_fork():
    globals()['cq']      = cq
    globals()['trimesh'] = trimesh
    globals()['cKDTree'] = cKDTree
    globals()['compute_normals_metrics'] = compute_normals_metrics


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except:
        pass


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd


def compute_metrics(pred_mesh, gt_mesh):
    return compute_cd(pred_mesh, gt_mesh), compute_iou(pred_mesh, gt_mesh)



def transform_real_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    mesh.apply_scale(2.0 / max(mesh.extents))  # Normalize to [-1, 1]
    return mesh

def transform_gt_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh



def transform_pred_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_scale(1.0 / 200)  # Normalize to [0, 1]
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def code_to_mesh_and_brep_less_safe(code_str, mesh_path, brep_path):
    safe_ns = {"cq": cq}
    ns=safe_ns.copy()
    #print(f"Executing code {code_str}")
    try:
        exec(code_str, ns)
        mesh = compound_to_mesh(ns["r"].val())
        # export files if needed
        # mesh.export(mesh_path)
        return mesh
    except Exception as e:
        print(f"Error executing CadQuery code : {e}")
        return None


def get_metrics_from_single_text(text, gt_file, pred_mesh_path, pred_brep_path, n_points):

    gt_file = os.path.abspath(gt_file)
    pred_mesh_dir = os.path.abspath(pred_mesh_path)
    pred_brep_dir = os.path.abspath(pred_brep_path)

    base_file = os.path.basename(gt_file).rsplit('.stl', 1)[0]

    #print(f"computing metrics for file: {gt_file}", flush=True)
    #print(f"saving temp mesh to: {pred_mesh_dir}", flush=True)

    mesh_path = os.path.abspath(os.path.join(pred_mesh_dir, base_file + '.stl'))
    brep_path = os.path.abspath(os.path.join(pred_brep_dir, base_file + '.step'))
    
    #t_cad = time.perf_counter()
    try:
        pred_mesh = code_to_mesh_and_brep_less_safe(text, mesh_path, brep_path)
    except Exception as e:
        return dict(file_name=base_file, cd=None, iou=None, auc=None, mean_cos=None)
    #print(f"[TIME] cad_exec: {time.perf_counter()-t_cad:.3f}s on worker pid={os.getpid()}")

    if pred_mesh is None:
        print("Skipping metrics: invalid prediction", flush=True)
        return dict(file_name=base_file, cd=None, iou=None, auc=None, mean_cos=None)
    #t_met = time.perf_counter()
    cd, iou, auc, mean_cos = None, None, None, None
    try:  # apply_transform fails for some reason; or mesh path can not exist
        gt_mesh = trimesh.load_mesh(gt_file)

        gt_mesh = transform_gt_mesh(gt_mesh)
        
        #print("Loaded and normalized ground truth", flush=True)
        
        pred_mesh = transform_pred_mesh(pred_mesh)
        #print("Normalizing prediction", flush=True)

        

        cd = compute_cd(gt_mesh, pred_mesh, n_points)

        iou = compute_iou(gt_mesh, pred_mesh)
        auc, mean_cos, _ = compute_normals_metrics(
                pred_mesh, gt_mesh, tol=2
            )
        #print(f"CD {cd} IoU {iou} AUC {auc} Mean Cos {mean_cos}", flush=True)

    except Exception as e:
        print(f"error for {base_file}: {e}", flush=True)
        pass
    #print(f"[TIME] metric computation without cadquery: {time.perf_counter()-t_cad:.3f}s on worker pid={os.getpid()}")
    del gt_mesh, pred_mesh
    return dict(file_name=base_file, cd=cd, iou=iou, auc=auc, mean_cos=mean_cos)




POOL = None

def init_pool(max_workers):
    print("Initializing POOOL", flush=True)
    global POOL
    if POOL is None:
        from multiprocessing import get_context
        ctx = get_context("forkserver")
        POOL = NonDaemonPool(
            processes=max_workers,
            initializer=init_worker,
            context=ctx
        )
        print("POOL Initialized", flush=True)

        import atexit
        atexit.register(lambda: (POOL.close(), POOL.join()))



def get_metrics_from_texts(texts, meshes, max_workers= None):
    print(f"[POOL] POOL size={POOL._processes} pid={os.getpid()}")
    t0 = time.perf_counter()
    
    temp_path = "./tmp_data"
    pred_mesh_path = os.path.join(temp_path, 'tmp_mesh')
    pred_brep_path = os.path.join(temp_path, 'tmp_brep')

    os.makedirs(pred_mesh_path, exist_ok=True)
    os.makedirs(pred_brep_path, exist_ok=True)

    n_points = 8192
    args = [
        (text, gt, pred_mesh_path, pred_brep_path, n_points)
        for text, gt in zip(texts, meshes)
    ]
    async_results = [POOL.apply_async(get_metrics_from_single_text, args=arg) for arg in args]
    results = []
    for res in async_results:
        try:
            results.append(res.get(timeout=60))
        except TimeoutError:
            print(f"[TIMEOUT] metrics task exceeded {60}s, skipping", flush=True)
            results.append(dict(file_name=None, cd=None, iou=None, auc=None, mean_cos=None))

    wait = time.perf_counter() - t0 
    print(f"TIME to get metrics for {len(texts)} samples : {wait}")

    return results
########## evaluate functions ##########


def code_to_mesh_and_brep(code_str, mesh_path, brep_path):

    #print(f"executing code {code_str}")
    # saves mesh and brep files from code string 
    try:
        ns = safe_ns.copy()
        exec(code_str, ns)
        compound = ns['r'].val()
        mesh = compound_to_mesh(compound)
        assert len(mesh.faces) > 2
        mesh.export(mesh_path)
        print("mesh exported successfully")
        # cq.exporters.export(compound, brep_path)
    except Exception as e:
        print("error executing the python code and exporting the mesh:", e)
        return