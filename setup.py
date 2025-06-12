from setuptools import setup

package_name = 'macadamia_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='SLAM-aware macadamia nut detector with row navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'explorer_node = macadamia_explorer.explorer_node:main',
        ],
    },
)
