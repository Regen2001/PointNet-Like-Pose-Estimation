# PointNet-Like-Pose-Estimation

This is a Final Year Project (FYP) in of School of Advanced Technology of Xi'an Jiaotong-Liverpool University. This FYP tried to realize the object pose estimation for robot grasping based on the point cloud. Therefore, we applied the PointNet network, which is an end-to-end framework, to the basis of our work.

As the time and hardware limitations, we only designed and trained our PointNet network and PoinetNet-like network, and given the Python functions that were used to realize point cloud collection and pre-procession, but not design the program that realizes the total process includes point cloud collection, point cloud pre-procession, objection classification, object pose estimation, and robot grasping. However, the flowchart for this program is shown as


```mermaid
graph TD
open[Open the camera] --> collect[Collect depth and color image]
collect --> visualize[Visualize RGB-D image]
visualize--> definite{Definite image}
definite --No--> collect
definite --Yes--> generate[Generate point cloud]
generate --> cut[Cut point clouds by distance]
generate --> monoitor(Real-time visualize RGB-D image)
cut --> delete[Delete plane]
delete --> outlier[Delete outlier]
outlier --> cluster[Cluster points]
cluster --> number{One or more than one cluster}
number --No--> collect
number --Yes--> pointnet[PointNet network: classify objection]
pointnet --> max[Max points and classes of itmes]
max --> pointnet-like[PontNet-like network: estimate pose]
pointnet-like --> grasp[Grasp objection]
grasp --> exit{Exit the program}
exit --No--> collect
exit --Yes--> close[Close the camera]
```
