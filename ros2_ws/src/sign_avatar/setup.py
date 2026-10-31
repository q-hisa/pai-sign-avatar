from glob import glob
from setuptools import setup

package_name = 'sign_avatar'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'planner = sign_avatar.planner_node:main',
            'player = sign_avatar.player_node:main',
            'sim = sign_avatar.sim_node:main',
            'recorder = sign_avatar.recorder_node:main',
        ],
    },
)
