import numpy as np

def lookAt(center, target, up):
    # code refer https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
    # generate w2c matrix
    f = (target - center); f = f/np.linalg.norm(f)
    s = np.cross(f, up); s = s/np.linalg.norm(s)
    u = np.cross(s, f); u = u/np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = s
    m[1, :-1] = u
    m[2, :-1] = -f
    m[-1, -1] = 1.0

    return m


def rot_from_init(dir, up):
    # input dir and up of a camera pose, output the rotation matrix
    dir_init = np.array([0, 0, 1])
    up_init = np.array([0, -1, 0])
    center = np.array([0, 0, 0])
    w2c_1 = lookAt(center, dir_init, up_init)

    dir = dir / np.linalg.norm(dir)
    up = up / np.linalg.norm(up)
    w2c_2 = lookAt(center, dir, up)

    R1 = w2c_1[:3, :3]
    R2 = w2c_2[:3, :3]
    R = R2.T @ R1
    return R


def _test_rot_from_init():
    dir = np.random.rand(3)
    dir = dir / np.linalg.norm(dir)
    up = np.random.rand(3)
    up = up - np.dot(up, dir) * dir  # make up orthogonal to dir
    up = up / np.linalg.norm(up)  # normalize up

    R = rot_from_init(dir, up)
    print(R)
    print()
    dir_init = np.array([0, 0, 1])
    up_init = np.array([0, -1, 0])
    print(dir_init @ R.T, dir)
    print(np.allclose(dir_init @ R.T, dir))
    print()
    print(up_init @ R.T, up)
    print(np.allclose(up_init @ R.T, up))


def generate_c2w_from_pnts(p1, p2):
    dir = p2 - p1
    dir = dir / np.linalg.norm(dir)

    m = np.eye(4)
    up = np.array([0, 0, 1])
    R = rot_from_init(dir, up)
    m[:3, :3] = R
    m[:3, 3] = p1
    return m


def interp_poses(poses, step):
    res = []
    for i in range(1, len(poses)):
        for t in range(step):
            res.append(poses[i - 1] + t * 1. / step * (poses[i] - poses[i - 1]))
    return res