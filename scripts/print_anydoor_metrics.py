import os
scene_path = "/media/data/yxy/ed-nerf/logs/tensoir/log_scannet/paper_01/"
scene_list = os.listdir(scene_path)
for scene in scene_list:
    metrics_file = os.path.join(scene_path, scene, scene, "imgs_test_all", "mean.txt")
    if not os.path.exists(metrics_file):
        continue
    with open(metrics_file, "r") as f:
        lines = f.readlines()
        print(scene)
        print(lines[0].strip())
        print(lines[1].strip())
        print(lines[2].strip())
        print(lines[3].strip())
        print("=====================================")
