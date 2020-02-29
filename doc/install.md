## Install
- install fvcore
    ```shell
    pip install -U 'git+https://github.com/facebookresearch/fvcore'
    ```
- run setup.py, open $project_root/$
  ```shell
    pip install -e.
    or
    pip install -e .
  ```
## Common Installation Issues

- ERROR: Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.

  - you can uninstall pip and install pip <10.0 and uninstall PyYAML

    ```shell
    python -m pip install --upgrade pip==9.0.3
    pip uninstall PyYAML
    ```


  - finish, upgrad pip

    ```shell
    python -m pip install --upgrade pip
    ```

    ​




## Output

- yolov3-darknet53
  - fpn
    - p3: $N \times 128 \times W/8 \times  H/8$
    - p4: $N\times 256 \times W/16 \times H/16​$
    - p5: $N\times 512\times W/32\times H/32$



- All boxes format

  ```python
  Boxes
  ```

  ​