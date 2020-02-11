## Install
- install fvcore
    ```shell
    pip install -U 'git+https://github.com/facebookresearch/fvcore'
    ```
- run setup.py, open $root_dir/$
  ```shell
    pip install -e.
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

