# train:

## launch gazebo enviroment

To launch the Gazebo model with the specified world, use the following command:

```bash
source ./devel/setup.bash

roslaunch full_vehicle_description gazebo_model_tow_no_rod.launch world_name:='$(find full_vehicle_description)/hospital/test_cam_lidar.world'
```

## start imitation learning

```bash
source ./devel/setup.bash

rosrun rl_enviroment train_lidar.py
```

## start rl finetune learning

1. setup the path of model parametes from imitation leaning stage
2. launch rl
```bash
source ./devel/setup.bash

rosrun rl_enviroment train_lidar_finetune.py
```

## test

```bash
source ./devel/setup.bash

rosrun rl_enviroment train_lidar_finetune.py --test True
```



