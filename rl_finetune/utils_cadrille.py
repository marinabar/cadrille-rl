import os
import time

os.environ["PYGLET_HEADLESS"] = "True"

from multiprocessing.pool import Pool
from multiprocessing import TimeoutError, Process
from multiprocessing import get_context
import numpy as np

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

# process initializer used in case of forkserver
def init_worker():
    os.environ["OMP_NUM_THREADS"]       = "1"
    os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    os.environ["MKL_NUM_THREADS"]       = "1"
    """
    import trimesh
    from scipy.spatial import cKDTree
    import cadquery as cq

    # make them available to your metric code
    globals()['trimesh'] = trimesh
    globals()['cKDTree'] = cKDTree
    globals()['cq'] = cq
    globals()['compute_normals_metrics'] = compute_normals_metrics"""


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


def code_to_mesh_and_brep_less_safe(code_str):
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


def get_metrics_from_single_text(text, gt_file, n_points, normalize="fixed"):

    gt_file = os.path.abspath(gt_file)
    base_file = os.path.basename(gt_file).rsplit('.stl', 1)[0]

    #print(f"computing metrics for file: {gt_file}", flush=True)
    
    #t_cad = time.perf_counter()
    try:
        # execute cadquery code
        pred_mesh = code_to_mesh_and_brep_less_safe(text)
    except Exception as e:
        return dict(file_name=base_file, cd=None, iou=None, auc=None)
    #print(f"[TIME] cad_exec: {time.perf_counter()-t_cad:.3f}s on worker pid={os.getpid()}")

    if pred_mesh is None:
        #print("Skipping metrics: invalid prediction", flush=True)
        return dict(file_name=base_file, cd=None, iou=None, auc=None)
    #t_met = time.perf_counter()
    cd, iou, auc = None, None, None
    try: 
        gt_mesh = trimesh.load_mesh(gt_file)

        if normalize == "fixed":
            gt_mesh = transform_gt_mesh(gt_mesh)
            pred_mesh = transform_pred_mesh(pred_mesh)
        else :
            gt_mesh = transform_real_mesh(gt_mesh)
            pred_mesh = transform_real_mesh(pred_mesh)

        cd = compute_cd(gt_mesh, pred_mesh, n_points)

        iou = compute_iou(gt_mesh, pred_mesh)

        #auc, _, _ = compute_normals_metrics(gt_mesh, pred_mesh, n_points=n_points, tol=2)

    except Exception as e:
        print(f"error for {base_file}: {e}", flush=True)
        pass

    #print(f"[TIME] metric computation without cadquery: {time.perf_counter()-t_cad:.3f}s on worker pid={os.getpid()}")
    """"
    finally:
        try:
            if gt_mesh is not None:
                del gt_mesh
            if pred_mesh is not None:
                del pred_mesh
        except:
            pass"""
    return dict(file_name=base_file, cd=cd, iou=iou, auc=auc)




POOL = None

def init_pool(max_workers):
    print("Initializing POOL", flush=True)
    global POOL
    if POOL is None:
        ctx = get_context("spawn")
        POOL = NonDaemonPool(
            processes=max_workers,
            initializer=init_worker,
            context=ctx,
        )
        #print("POOL Initialized", flush=True)
    return POOL

def close_pool():
    global POOL
    if POOL is not None:
        POOL.close()
        POOL.join()
        POOL = None


def timed_process_text(arg, timeout=60):
    """
    Supervisor function that *workers* will call.
    Spawns one child Process for real work, kills it on timeout.
    """
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result  


def _run_child(conn, arg):
    """Executed in a fresh subprocess."""
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)          # may be None or a string
    finally:
        conn.close()


def get_metrics_from_texts(texts, meshes, max_workers= None, normalize="fixed"):
    print(f"[POOL] POOL size={POOL._processes} pid={os.getpid()}")
    #t0 = time.perf_counter()
    #print(f"Example of one generated code : {texts[0]}")
    n_points = 8192
    args = [
        (text, gt, n_points, normalize)
        for text, gt in zip(texts, meshes)
    ]
    async_results = [POOL.apply_async(timed_process_text, args=(arg,)) for arg in args]
    results = []
    for res in async_results:
        output = res.get()
        if output == "__TIMEOUT__" or output == "__CRASH__":
            print(f"[{output}] metrics task computation ERROR, skipping", flush=True)
            results.append(dict(file_name=None, cd=None, iou=None, auc=None))
        else:
            results.append(output)

    #wait = time.perf_counter() - t0 
    #print(f"TIME to get metrics for {len(texts)} samples : {wait}")

    return results

def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area under the curve (AUC) of the angle distribution between the normals.
    Returns the aoc and mean_cos_sim
    """
    #tol = 0.01 * max(gt_mesh.extents.max(), pred_mesh.extents.max())  # 1% of the mesh extent
    tol = pred_mesh.extents.max() * tol  / 100

    gt_points, gt_face_indexes = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh.sample.sample_surface(pred_mesh, n_points)

    # normals of sampled points
    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]

    tree = cKDTree(pred_points)
    neighbors = tree.query_ball_point(gt_points, r=tol)
    # get the indices of the neighbors for each ground-truth point

    valid_pred_normals = []
    valid_gt_normals = []
    valid_gt_points = []
    valid_pred_points = []

    for i, idxs in enumerate(neighbors):
        if len(idxs) == 0:
            continue
        gn = gt_normals[i]
        pn_neighbors = pred_normals[idxs] # candidates

        valid_gt_normals.append(gn)
        dots = (pn_neighbors * gn).sum(axis=1)  # (k,)
        best_idx = np.argmax(dots)  # index of the best aligned normal

        valid_pred_normals.append(pn_neighbors[best_idx])  # (3,)

        valid_gt_points.append(gt_points[i])  # (3,)
        valid_pred_points.append(pred_points[idxs[best_idx]])  # (3,)

    if len(valid_pred_normals) == 0:
        return 1.0, 0.0, 100

    valid_gt_normals = np.vstack(valid_gt_normals)
    valid_pred_normals = np.vstack(valid_pred_normals)
    valid_gt_points = np.vstack(valid_gt_points)
    valid_pred_points = np.vstack(valid_pred_points)

    nb_invalid = n_points - len(valid_pred_normals)
    per_invalid = nb_invalid / n_points * 100
    #print(f"Number of points with no neighbors within tol: {nb_invalid} out of {n_points} ({per_invalid:.2f}%)")
    
    
    # compute cosine similarity
    cos_sim = (valid_pred_normals * valid_gt_normals).sum(axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    mean_cos_sim = np.mean(cos_sim)
    
    # distribution of angles between normals
    angles = np.arccos(cos_sim)
    angles = np.sort(angles)

    # add invalid points to the end of the array with max angle (pi)
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))

    N = len(angles)
    cdf = np.arange(1, N+1) / N

    from numpy import trapz
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0],   cdf,   [1.0]))
    auc_normalized = trapz(y, x) / np.pi  # Normalize by the maximum possible aoc (which is pi)

    #we want to maximize the AUC
    #aoc_normalized = 1 - auc_normalized
    # plot the aoc
    #if aoc_normalized > 0.3:
        #print(f"HIGH aoc: {aoc_normalized:.2f}")
        #plot_aoc(angles, cdf, title='aoc of Normal Angles', aoc_value=aoc_normalized)

    #print(f"AUC {auc_normalized}", flush=True)
    return auc_normalized, mean_cos_sim, per_invalid