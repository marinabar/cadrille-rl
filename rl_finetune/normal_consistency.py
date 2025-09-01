def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area over the curve (AOC) of the angle distribution between the normals.
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


    if visualize:
        save_normals(valid_pred_normals, valid_gt_normals, valid_pred_points, valid_gt_points)


    return auc_normalized, mean_cos_sim, per_invalid

def save_normals(pred_normals, gt_normals, pred_points, gt_points):
    import os
    import matplotlib.pyplot as plt
    os.makedirs('plots', exist_ok=True)

    # get points only with highest value along z axis
    pred_normals = pred_normals[pred_points[:, 2].argsort()[-50:]]
    gt_normals = gt_normals[gt_points[:, 2].argsort()[-50:]]
    pred_points = pred_points[pred_points[:, 2].argsort()[-50:]]
    gt_points = gt_points[gt_points[:, 2].argsort()[-50:]]


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    sample_size = 50
    # get 50 points with lowest cosine similarity
    cos_sim = (pred_normals * gt_normals).sum(axis=1)
    indices = np.argsort(cos_sim)[:sample_size]
    pred_normals = pred_normals[indices]
    gt_normals = gt_normals[indices]
    pred_points = pred_points[indices]
    gt_points = gt_points[indices]

    ax.set_box_aspect((1,1,1))

    ax.quiver(pred_points[:sample_size, 0], pred_points[:sample_size, 1], pred_points[:sample_size, 2],
              pred_normals[:sample_size, 0], pred_normals[:sample_size, 1], pred_normals[:sample_size, 2],
              length=0.1, color='r', label='Predicted Normals', alpha=0.5, normalize=True)

    ax.quiver(gt_points[:sample_size, 0], gt_points[:sample_size, 1], gt_points[:sample_size, 2],
              gt_normals[:sample_size, 0], gt_normals[:sample_size, 1], gt_normals[:sample_size, 2],
              length=0.1, color='b', label='Ground Truth Normals', alpha=0.5, normalize=True)

    ax.set_title('Normals Visualization')
    ax.legend()
    
    fig.savefig('plots/normals_vis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)